import os
import re
import time
import math
import json
import sqlite3
import hashlib
import secrets
import unicodedata
from collections import OrderedDict
from datetime import datetime, timedelta
from typing import Optional, List

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from passlib.context import CryptContext
from jose import JWTError, jwt
from groq import Groq
from dotenv import load_dotenv

# ─────────────────────────── Config ───────────────────────────
load_dotenv()
SECRET_KEY = secrets.token_hex(32)
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24h
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
KB_PATH = os.path.join(os.path.dirname(__file__), "Knowledge_Bank.txt")
DB_PATH = os.path.join(os.path.dirname(__file__), "users.db")
TOP_K = 5
MIN_RETRIEVAL_SCORE = 0.10  # hybrid scores are generally lower; tuned for BM25+dense
BM25_WEIGHT = 0.4           # α for hybrid: score = α*bm25 + (1-α)*dense
DENSE_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"  # 118MB, native Bengali support
CONTEXT_WINDOW = 5  # last N turns kept for context
BM25_GATE_SCORE = 0.65  # if BM25 is strong, skip dense for speed
DENSE_RERANK_TOP_N = 60  # only rerank top-N BM25 candidates with dense
QUERY_EMBED_CACHE_SIZE = 512
SEMANTIC_CONTEXT_MIN_SIM = 0.35  # cosine similarity threshold for context reuse
RETRIEVAL_CONFIDENCE_MARGIN = 0.04  # top1 - top2 threshold
PRODUCT_QUERY_SIM_THRESHOLD = 0.45
PRODUCT_NAME_BIND_THRESHOLD = 0.60
PRODUCT_NAME_MARGIN = 0.04
DOMAIN_CLASSIFY_MARGIN = 0.02
STORE_QUERY_ANCHOR = "আমি দোকানের পণ্য, দাম, রিভিউ, রেটিং, ডেলিভারি, ওয়ারেন্টি, স্টক বা ছাড় সম্পর্কে জানতে চাই।"
GENERAL_CHAT_ANCHOR = "আমি সাধারণ কথা বলছি বা সাধারণ প্রশ্ন করছি।"
FORBIDDEN_PHRASES = {
    "discount": [
        "discount",
        "special price",
        "ডিসকাউন্ট",
        "ছাড়",
        "ছাড়",
        "বিশেষ মূল্য",
        "বিশেষ মূল্য ছাড়",
        "বিশেষ মূল্য ছাড়",
    ],
    "gift_wrap": [
        "gift wrap",
        "gift wrapping",
        "গিফট র্যাপ",
        "গিফট র্যাপিং",
        "উপহার মোড়ক",
        "উপহার মোড়ক",
    ],
}

# ─────────────────────────── Text Normalization ───────────────────────────
_BN_DIGIT_MAP = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")

def _looks_spaced_bengali(text: str) -> bool:
    # Detect very short, letter-spaced Bengali queries like "দা ম ক ত"
    tokens = re.findall(r'[\u0980-\u09FF]+|[a-zA-Z0-9]+', text)
    if not tokens:
        return False
    short = sum(1 for t in tokens if len(t) <= 2)
    return (len(tokens) <= 5 and short == len(tokens)) or (len(tokens) <= 6 and short / len(tokens) >= 0.8)

def normalize_text(text: str) -> str:
    # Unicode normalization + cleanup for Bengali/English mixed text
    text = unicodedata.normalize("NFKC", text)
    # Remove zero-width characters
    text = text.replace("\u200c", "").replace("\u200d", "").replace("\ufeff", "")
    # Normalize Bengali letter YA variants
    text = text.replace("\u09DF", "\u09AF\u09BC")  # য় -> য়
    # Normalize Bengali precomposed variants to base+Nukta forms
    text = text.replace("\u09DC", "\u09A1\u09BC")  # ড় -> ড়
    text = text.replace("\u09DD", "\u09A2\u09BC")  # ঢ় -> ঢ়
    # Normalize danda variations
    text = text.replace("৷", "।")
    # Normalize Bengali digits to ASCII digits
    text = text.translate(_BN_DIGIT_MAP)
    # Collapse spaces between Bengali letters only for "spaced-out" queries
    if _looks_spaced_bengali(text):
        text = re.sub(r'([\u0980-\u09FF])\s+(?=[\u0980-\u09FF])', r'\1', text)
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text

def truncate_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0].strip()

def extract_product_facts(rag, product_name: str) -> dict:
    docs = rag.docs_for_product(product_name)
    facts = {
        "has_discount": False,
        "discount_sentence": "",
        "has_money_amount": False,
        "has_percent": False,
        "warranty_brands": set(),
        "fast_delivery": False,
        "home_delivery": False,
        "payment_methods": set(),
        "rating": "",
        "popular": False,
        "best_selling": False,
        "feature_sentence": "",
    }

    discount_phrases = ["ছাড়", "ছাড়", "বিশেষ অফার", "বিশেষ মূল্য", "discount"]
    for doc in docs:
        if "দ্রুত ডেলিভারি" in doc:
            facts["fast_delivery"] = True
        if "হোম ডেলিভারি" in doc:
            facts["home_delivery"] = True
        if "ক্যাশ অন ডেলিভারি" in doc:
            facts["payment_methods"].add("ক্যাশ অন ডেলিভারি")
        if "বিকাশ" in doc:
            facts["payment_methods"].add("বিকাশ")
        if "নগদ" in doc:
            facts["payment_methods"].add("নগদ")
        if "কার্ড" in doc:
            facts["payment_methods"].add("কার্ড")
        if "জনপ্রিয়" in doc or "বহুল ব্যবহৃত" in doc:
            facts["popular"] = True
        if "সবচেয়ে বেশি বিক্রিত" in doc or "সবচেয়ে বেশি বিক্রিত পণ্যগুলোর একটি" in doc:
            facts["best_selling"] = True

        # Rating
        m = re.search(r"(\d+(?:\.\d+)?)\s*স্টার", doc)
        if m and not facts["rating"]:
            facts["rating"] = m.group(1)

        # Warranty brands
        for m in re.finditer(r"এই পণ্যের\s+([^\s]+)\s+ওয়ারেন্টি", doc):
            facts["warranty_brands"].add(m.group(1))
        for m in re.finditer(r"([^\s]+)\s+ওয়ারেন্টি", doc):
            facts["warranty_brands"].add(m.group(1))

        # Discount sentence
        if not facts["discount_sentence"]:
            for s in doc.split("।"):
                s = s.strip()
                if s and any(p in s for p in discount_phrases):
                    facts["discount_sentence"] = truncate_text(s, 140)
                    facts["has_discount"] = True
                    facts["has_money_amount"] = bool(re.search(r"\d+\s*টাকা", s))
                    facts["has_percent"] = bool(re.search(r"\d+\s*%|\d+\s*শতাংশ", s))
                    break

        # Feature sentence: pick the 2nd sentence if reasonable and not discount/gift
        if not facts["feature_sentence"]:
            sentences = [t.strip() for t in doc.split("।") if t.strip()]
            if len(sentences) > 1:
                cand = sentences[1]
                if not any(p in cand for p in discount_phrases + ["গিফট র্যাপ", "উপহার"]):
                    facts["feature_sentence"] = truncate_text(cand, 140)

    return facts

# Use PBKDF2-SHA256 to avoid bcrypt's 72-byte limit.
pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/token")
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

# ─────────────────────────── Database ───────────────────────────
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            hashed_password TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        );
        CREATE TABLE IF NOT EXISTS user_state (
            user_id INTEGER PRIMARY KEY,
            last_product TEXT,
            last_intent TEXT,
            last_product_known INTEGER DEFAULT 1,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        );
    """)
    # Lightweight migration for existing DBs
    try:
        conn.execute("ALTER TABLE user_state ADD COLUMN last_intent TEXT")
    except sqlite3.OperationalError:
        pass
    try:
        conn.execute("ALTER TABLE user_state ADD COLUMN last_product_known INTEGER DEFAULT 1")
    except sqlite3.OperationalError:
        pass
    conn.commit()
    conn.close()

def get_user_state(user_id: int) -> tuple[Optional[str], Optional[str], bool]:
    conn = get_db()
    row = conn.execute(
        "SELECT last_product, last_intent, last_product_known FROM user_state WHERE user_id=?",
        (user_id,)
    ).fetchone()
    conn.close()
    if not row:
        return None, None, True
    last_product = row["last_product"] if row["last_product"] else None
    last_intent = row["last_intent"] if row["last_intent"] else None
    known = True
    try:
        if row["last_product_known"] is not None:
            known = bool(int(row["last_product_known"]))
    except Exception:
        known = True
    # Backward compatibility if old data used __UNK__ prefix
    if last_product and last_product.startswith("__UNK__"):
        return last_product[len("__UNK__"):], last_intent, False
    return last_product, last_intent, known

def set_user_state(
    user_id: int,
    product_name: Optional[str] = None,
    intent: Optional[str] = None,
    product_known: Optional[bool] = None,
):
    conn = get_db()
    conn.execute(
        """
        INSERT INTO user_state (user_id, last_product, last_intent, last_product_known, updated_at)
        VALUES (?,?,?,?,CURRENT_TIMESTAMP)
        ON CONFLICT(user_id) DO UPDATE SET
            last_product=COALESCE(excluded.last_product, user_state.last_product),
            last_intent=COALESCE(excluded.last_intent, user_state.last_intent),
            last_product_known=COALESCE(excluded.last_product_known, user_state.last_product_known),
            updated_at=CURRENT_TIMESTAMP
        """,
        (user_id, product_name, intent, None if product_known is None else (1 if product_known else 0))
    )
    conn.commit()
    conn.close()

# ─────────────────────────── BM25 + Dense Hybrid RAG Engine ───────────────────────────
class RAGEngine:
    _STOP_TOKENS = {"ও", "এবং", "অথবা", "and", "or", "the", "a", "an", "of", "for"}

    def __init__(self, kb_path: str):
        self.documents: List[str] = []
        self.product_names: set[str] = set()
        self.product_name_tokens: List[List[str]] = []
        self.product_docs: dict[str, List[int]] = {}
        self.product_token_map: dict[str, List[str]] = {}
        # BM25
        self.bm25: Optional[BM25Okapi] = None
        # Dense
        self._model = SentenceTransformer(DENSE_MODEL)
        self.doc_embeddings: Optional[np.ndarray] = None
        self._q_cache: "OrderedDict[str, np.ndarray]" = OrderedDict()
        self.product_name_list: List[str] = []
        self.product_name_embeddings: Optional[np.ndarray] = None
        self._store_anchor_vec: Optional[np.ndarray] = None
        self._general_anchor_vec: Optional[np.ndarray] = None
        self._load_and_index(kb_path)

    def _normalize_text(self, text: str) -> str:
        return normalize_text(text)

    def _tokenize(self, text: str) -> List[str]:
        text = self._normalize_text(text).lower()
        return re.findall(r'[\u0980-\u09FF]+|[a-zA-Z0-9]+', text)

    def _load_and_index(self, path: str):
        t0 = time.time()
        with open(path, encoding="utf-8") as f:
            raw = f.read().replace("\r\n", "\n")

        # Split before normalization so we don't collapse paragraph boundaries
        paras_raw = [p.strip() for p in re.split(r'\n\s*\n', raw) if p.strip()]
        paras = [self._normalize_text(p) for p in paras_raw]
        self.documents = paras

        # Extract product names from first sentence of each paragraph
        doc_tokens = []
        for p in paras:
            first_sentence = p.split("।", 1)[0].strip()
            if first_sentence:
                name_norm = self._normalize_text(first_sentence).lower()
                self.product_names.add(name_norm)
                name_tokens = self._tokenize(first_sentence)
                self.product_name_tokens.append(name_tokens)
                idx = len(doc_tokens)
                self.product_docs.setdefault(name_norm, []).append(idx)
                # Build token->product map for single-token matches
                for tok in name_tokens:
                    if tok in self._STOP_TOKENS or len(tok) < 2:
                        continue
                    self.product_token_map.setdefault(tok, []).append(name_norm)
            tokens = self._tokenize(p)
            doc_tokens.append(tokens)

        # BM25 index
        self.bm25 = BM25Okapi(doc_tokens)

        # Dense embeddings — load from cache if KB hasn't changed, else encode & save
        cache_path = path + ".embeddings.v5.npy"
        kb_mtime = os.path.getmtime(path)
        if os.path.exists(cache_path) and os.path.getmtime(cache_path) >= kb_mtime:
            print(f"[RAG] Loading cached embeddings from {cache_path}")
            self.doc_embeddings = np.load(cache_path)
        else:
            print(f"[RAG] Encoding {len(paras)} documents with dense model (first run)...")
            self.doc_embeddings = self._model.encode(
                paras,
                batch_size=64,
                normalize_embeddings=True,
                show_progress_bar=True,
            ).astype(np.float32)
            np.save(cache_path, self.doc_embeddings)
            print(f"[RAG] Embeddings cached to {cache_path}")

        # Product-name embeddings (unique names)
        self.product_name_list = sorted(self.product_docs.keys())
        if self.product_name_list:
            self.product_name_embeddings = self._model.encode(
                self.product_name_list,
                normalize_embeddings=True,
                batch_size=128,
                show_progress_bar=False,
            ).astype(np.float32)

        # Domain anchors for simple semantic classification
        self._store_anchor_vec = self._model.encode(
            STORE_QUERY_ANCHOR, normalize_embeddings=True
        ).astype(np.float32)
        self._general_anchor_vec = self._model.encode(
            GENERAL_CHAT_ANCHOR, normalize_embeddings=True
        ).astype(np.float32)

        print(f"[RAG] Indexed {len(paras)} docs in {(time.time()-t0)*1000:.1f}ms")

    def _bm25_scores_normalised(self, query_tokens: List[str]) -> np.ndarray:
        scores = self.bm25.get_scores(query_tokens).astype(np.float32)
        max_s = scores.max()
        if max_s > 0:
            scores /= max_s
        return scores

    def _dense_scores(self, query: str) -> np.ndarray:
        qvec = self._get_query_embedding(query)
        return self.doc_embeddings @ qvec  # cosine similarity

    def _get_query_embedding(self, query: str) -> np.ndarray:
        key = self._normalize_text(query).lower()
        cached = self._q_cache.get(key)
        if cached is not None:
            # Move to end (LRU)
            self._q_cache.move_to_end(key)
            return cached
        vec = self._model.encode(key, normalize_embeddings=True).astype(np.float32)
        self._q_cache[key] = vec
        if len(self._q_cache) > QUERY_EMBED_CACHE_SIZE:
            self._q_cache.popitem(last=False)
        return vec

    def retrieve(self, query: str, top_k: int = TOP_K) -> tuple[List[str], float, float]:
        t0 = time.time()
        tokens = self._tokenize(query)

        bm25_s = self._bm25_scores_normalised(tokens)
        # Fast path: if BM25 is confident, skip dense
        if bm25_s.size:
            k = min(top_k, bm25_s.size)
            top_bm25_idx = np.argpartition(bm25_s, -k)[-k:]
            top_bm25_idx = top_bm25_idx[np.argsort(bm25_s[top_bm25_idx])[::-1]]
            top_bm25_score = float(bm25_s[top_bm25_idx[0]])
            second_bm25_score = float(bm25_s[top_bm25_idx[1]]) if len(top_bm25_idx) > 1 else 0.0
            if top_bm25_score >= BM25_GATE_SCORE:
                results = [self.documents[i] for i in top_bm25_idx if bm25_s[i] > 0.01]
                print(f"[RAG] BM25-only retrieval {(time.time()-t0)*1000:.1f}ms  top_score={top_bm25_score:.4f}")
                return results, top_bm25_score, second_bm25_score

        # Dense rerank: only on top-N BM25 candidates for speed
        dense_s = None
        if DENSE_RERANK_TOP_N and bm25_s.size > 0:
            cand_n = min(DENSE_RERANK_TOP_N, bm25_s.size)
            cand_idx = np.argpartition(bm25_s, -cand_n)[-cand_n:]
            qvec = self._get_query_embedding(query)
            dense_s = self.doc_embeddings[cand_idx] @ qvec
            hybrid = BM25_WEIGHT * bm25_s[cand_idx] + (1 - BM25_WEIGHT) * dense_s
            order = np.argsort(hybrid)[::-1]
            top_score = float(hybrid[order[0]]) if len(order) else 0.0
            second_score = float(hybrid[order[1]]) if len(order) > 1 else 0.0
            results = []
            for j in order[:top_k]:
                if hybrid[j] > 0.01:
                    results.append(self.documents[cand_idx[j]])
            print(f"[RAG] Hybrid (rerank) {(time.time()-t0)*1000:.1f}ms  top_score={top_score:.4f}")
            return results, top_score, second_score

        # Full hybrid (fallback)
        dense_s = self._dense_scores(query)
        hybrid = BM25_WEIGHT * bm25_s + (1 - BM25_WEIGHT) * dense_s
        top_idx = np.argsort(hybrid)[::-1][:top_k]
        top_score = float(hybrid[top_idx[0]]) if len(top_idx) else 0.0
        second_score = float(hybrid[top_idx[1]]) if len(top_idx) > 1 else 0.0
        results = [self.documents[i] for i in top_idx if hybrid[i] > 0.01]
        print(f"[RAG] Hybrid retrieval {(time.time()-t0)*1000:.1f}ms  top_score={top_score:.4f}")
        return results, top_score, second_score

    def semantic_product_match(self, message: str) -> tuple[bool, float, Optional[str]]:
        if self.product_name_embeddings is None or not self.product_name_list:
            return False, 0.0, None
        qvec = self._get_query_embedding(message)
        sims = self.product_name_embeddings @ qvec
        best_idx = int(np.argmax(sims))
        best_score = float(sims[best_idx])
        best_name = self.product_name_list[best_idx]
        second_score = 0.0
        if len(sims) > 1:
            second_score = float(np.partition(sims, -2)[-2])
        confident = (best_score >= PRODUCT_QUERY_SIM_THRESHOLD) and ((best_score - second_score) >= PRODUCT_NAME_MARGIN)
        return confident, best_score, best_name

    def query_domain(self, message: str) -> str:
        if self._store_anchor_vec is None or self._general_anchor_vec is None:
            return "store"
        qvec = self._get_query_embedding(message)
        sim_store = float(qvec @ self._store_anchor_vec)
        sim_general = float(qvec @ self._general_anchor_vec)
        if sim_store >= sim_general + DOMAIN_CLASSIFY_MARGIN:
            return "store"
        return "general"


    # ── Product-name helpers (unchanged logic, shared _STOP_TOKENS) ──

    def _product_match_score(self, name_tokens: List[str], msg_tokens: set) -> float:
        name_set = set(t for t in name_tokens if t not in self._STOP_TOKENS)
        if not name_set:
            return 0.0
        overlap = len(name_set & msg_tokens)
        if len(name_set) <= 2:
            return 1.0 if name_set.issubset(msg_tokens) else 0.0
        if overlap < 2:
            return 0.0
        ratio = overlap / len(name_set)
        # Be a bit more permissive for longer product names
        if len(name_set) >= 4 and ratio >= 0.4:
            return ratio
        return ratio if ratio >= 0.5 else 0.0

    def matched_product_names(self, message: str) -> List[str]:
        """Return KB product names that match the message (used to hint the LLM)."""
        msg = self._normalize_text(message).lower()
        matched = []
        # Exact substring matches
        for name in self.product_names:
            if name in msg:
                matched.append(name)
        if matched:
            return matched
        # Token overlap matches
        msg_tokens = set(t for t in self._tokenize(msg) if t not in self._STOP_TOKENS)
        if not msg_tokens:
            return []
        for i, tokens in enumerate(self.product_name_tokens):
            if self._product_match_score(tokens, msg_tokens) >= 0.5:
                matched.append(" ".join(tokens))
        # Single-token fallback (e.g., "নুডুলস", "ঘড়ি")
        # Score each candidate product by how many query tokens it matches,
        # then return only the top-scoring ones to avoid false positives from
        # shared tokens like "তাজা" matching both "তাজা পাউরুটি" and "তাজা দই".
        if not matched:
            product_scores: dict[str, int] = {}
            for tok in msg_tokens:
                if tok in self.product_token_map:
                    # Use set() to avoid inflating scores with duplicate KB entries
                    for pname in set(self.product_token_map[tok]):
                        product_scores[pname] = product_scores.get(pname, 0) + 1
            if product_scores:
                max_score = max(product_scores.values())
                matched = [p for p, s in product_scores.items() if s == max_score]
        # Deduplicate while preserving order
        seen, out = set(), []
        for m in matched:
            if m not in seen:
                seen.add(m)
                out.append(m)
        return out

    def has_product_token(self, message: str) -> bool:
        msg_tokens = set(t for t in self._tokenize(message.lower()) if t not in self._STOP_TOKENS)
        return any(tok in self.product_token_map for tok in msg_tokens)

    def has_product_mention(self, message: str) -> bool:
        return len(self.matched_product_names(message)) > 0

    def has_unknown_product_qualifiers(self, message: str) -> bool:
        """
        Returns True when the message has KNOWN product tokens (in product_token_map)
        alongside UNKNOWN significant words (not in product_token_map, not stop/question words,
        length >= 3).  Detects brand-specific queries like 'নাইকি জুতা' where 'নাইকি' is
        a brand not in the KB — so we should say "not available" instead of matching the
        generic product.
        """
        _QUESTION_WORDS = {
            "কি", "কী", "কেমন", "কত", "আছে", "আছেন", "দাম", "রিভিউ",
            "রেটিং", "ওয়ারেন্টি", "ডেলিভারি", "ডেলিভারী", "পাওয়া", "যায়",
            "sell", "buy", "available", "কোথায়", "কেন", "হয়", "করেন",
            "বিক্রি", "কিনতে", "পারি", "পাব", "পাবেন", "কিনব", "দিন",
            "আপনি", "আপনার", "আপনাদের", "তুমি", "তোমার", "তোমরা",
            "আমি", "আমরা", "দোকান", "দোকানে", "স্টোর", "স্টোরে", "শপ", "শপে",
        }
        tokens = set(self._tokenize(message.lower()))
        has_known = any(t in self.product_token_map for t in tokens)
        if not has_known:
            return False
        has_unknown = any(
            t not in self.product_token_map
            and t not in self._STOP_TOKENS
            and t not in _QUESTION_WORDS
            and len(t) >= 3
            for t in tokens
        )
        return has_unknown

    def retrieve_by_product_name(self, query: str, top_k: int = TOP_K) -> List[str]:
        """Direct product-name lookup — fallback when hybrid score is still weak."""
        msg_tokens = set(t for t in self._tokenize(query.lower()) if t not in self._STOP_TOKENS)
        scored = [
            (self._product_match_score(tokens, msg_tokens), i)
            for i, tokens in enumerate(self.product_name_tokens)
        ]
        scored = [(s, i) for s, i in scored if s >= 0.5]
        scored.sort(reverse=True)
        return [self.documents[i] for _, i in scored[:top_k]]

    def docs_for_product(self, product_name: str) -> List[str]:
        key = self._normalize_text(product_name).lower()
        idxs = self.product_docs.get(key, [])
        return [self.documents[i] for i in idxs]


# ─────────────────────────── Context Manager ───────────────────────────
class ContextManager:
    """
    Detects if the user's message contains its own context.
    If not, injects recent conversation history into the RAG query.
    """

    _QUESTION_HINTS = [
        "কি", "কী", "কেমন", "কোথায়", "কেন", "কিভাবে", "কত", "দাম",
        "আছে", "রিভিউ", "রেটিং", "স্টার", "price", "review", "rating", "available"
    ]
    _NON_CONTEXT_TOKENS = {
        # Stop/connectors
        "ও", "এবং", "অথবা", "and", "or", "the", "a", "an", "of", "for",
        # Question/intent words
        "কি", "কী", "কেমন", "কোথায়", "কেন", "কিভাবে", "কত", "দাম", "মূল্য",
        "আছে", "রিভিউ", "রেটিং", "স্টার", "price", "review", "rating",
        "available", "availability", "স্টক", "stock", "পাওয়া", "পাওয়া", "যায়", "যায়", "যাবে",
        "কিনতে", "কিনব", "পারি", "পাব", "পাবেন", "বিক্রি", "sell", "buy",
        "ডেলিভারি", "ডেলিভারী", "delivery", "ওয়ারেন্টি", "ওয়ারেন্টি", "warranty",
        "ফিচার", "বৈশিষ্ট্য", "features", "discount", "অফার", "ছাড়", "ছাড়",
        "পেমেন্ট", "payment", "বিকাশ", "নগদ", "কার্ড",
        # Pronouns/determiners
        "আমি", "আমরা", "আপনি", "আপনার", "আপনাদের", "তুমি", "তোমার", "তোমরা",
        "সে", "তারা", "এটা", "ওটা", "এইটা", "ঐটা", "এই", "ঐ", "ওই", "এখানে", "ওখানে",
        "আগেরটা", "পরেরটা",
        # Greetings / fillers
        "আসসালামু", "আলাইকুম", "সালাম", "হাই", "হ্যালো", "ভাই", "বোন", "ধন্যবাদ",
        "কেমন", "আছেন", "আছো", "আছেন?", "জানাবেন",
        # Shop words
        "দোকান", "দোকানে", "স্টোর", "স্টোরে", "শপ", "শপে",
    }
    _CONJUNCTION_TOKENS = {"এবং", "ও", "অথবা", "and", "or", "&"}
    _GREETING_TOKENS = {
        "আসসালামু", "আলাইকুম", "সালাম", "হাই", "হ্যালো", "ভাই", "বোন",
        "ধন্যবাদ", "কেমন", "আছেন", "আছো", "শুভ", "সকাল", "বিকেল", "রাত",
    }

    def __init__(self, rag_engine: "RAGEngine"):
        self.rag = rag_engine

    def _has_product_context(self, message: str) -> bool:
        msg_norm = normalize_text(message)
        if self.rag.has_product_mention(msg_norm) or self.rag.has_product_token(msg_norm):
            return True
        # Heuristic: if there is any meaningful token left after removing
        # question/intent/greeting words, treat it as having its own context.
        tokens = self.rag._tokenize(msg_norm.lower())
        content_tokens = [
            t for t in tokens
            if t not in self._NON_CONTEXT_TOKENS
            and len(t) >= 2
            and not t.isdigit()
        ]
        return len(content_tokens) > 0

    def has_own_context(self, message: str) -> bool:
        return self._has_product_context(message)

    def extract_candidate_products(self, message: str) -> List[str]:
        msg_norm = normalize_text(message)
        tokens = self.rag._tokenize(msg_norm.lower())
        candidates = [
            t for t in tokens
            if t not in self._NON_CONTEXT_TOKENS
            and len(t) >= 2
            and not t.isdigit()
        ]
        # Deduplicate while preserving order
        seen, out = set(), []
        for c in candidates:
            if c not in seen:
                seen.add(c)
                out.append(c)
        return out

    def is_multi_product(self, message: str) -> bool:
        msg_norm = normalize_text(message)
        tokens = self.rag._tokenize(msg_norm.lower())
        has_conj = any(t in self._CONJUNCTION_TOKENS for t in tokens) or ("," in msg_norm)
        if not has_conj:
            return False
        candidates = self.extract_candidate_products(msg_norm)
        return len(candidates) >= 2

    def extract_recent_products_with_candidates(self, history: List[dict]) -> tuple[List[str], List[str], bool]:
        """
        Returns (known_products, candidate_products, has_multi) from the most recent user
        message that has its own context.
        """
        for h in reversed(history):
            if h.get("role") != "user":
                continue
            h_norm = normalize_text(h.get("content", ""))
            if not self._has_product_context(h_norm):
                continue
            candidates = self.extract_candidate_products(h_norm)
            names = self.rag.matched_product_names(h_norm)
            if names:
                names = [n.split(",")[0].strip() for n in names]
            has_multi = self.is_multi_product(h_norm)
            if candidates or names:
                return names, candidates, has_multi
        return [], [], False

    def _extract_recent_products(self, history: List[dict]) -> List[str]:
        # Find the most recent user message that mentions products
        for h in reversed(history):
            if h.get("role") != "user":
                continue
            h_norm = normalize_text(h.get("content", ""))
            names = self.rag.matched_product_names(h_norm)
            if names:
                # KB product names may include a comma + description (e.g. "তাজা পাউরুটি, প্রতিদিন...").
                # Keep only the part before the comma so context stays short and clean.
                return [n.split(",")[0].strip() for n in names]
        return []

    def _is_question_like(self, message: str) -> bool:
        msg_lower = normalize_text(message).lower()
        return ("?" in message) or any(h in msg_lower for h in self._QUESTION_HINTS)

    def is_greeting_only(self, message: str) -> bool:
        msg_norm = normalize_text(message)
        tokens = self.rag._tokenize(msg_norm.lower())
        if not tokens:
            return False
        non_greet = [t for t in tokens if t not in self._GREETING_TOKENS]
        # If all tokens are greetings/pleasantries, treat as greeting
        return len(non_greet) == 0

    def build_rag_query(
        self,
        message: str,
        history: List[dict],
        followup_intent: bool,
        has_followup_cue: bool,
    ) -> str:
        """
        Augment sparse queries with recent product context only when the
        message itself lacks product mentions.
        """
        if self._has_product_context(message):
            return message

        recent_products = self._extract_recent_products(history)
        if not recent_products:
            return message

        if followup_intent or has_followup_cue or self._is_question_like(message):
            return f"{' '.join(recent_products)} {message}"

        return message


# ─────────────────────────── Semantic Context Resolver ───────────────────────────
class SemanticContextResolver:
    """
    Uses semantic similarity to pick the most relevant recent user turn
    when the current query is underspecified.
    """
    def __init__(self, rag_engine: "RAGEngine"):
        self.rag = rag_engine

    def _similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(a @ b)  # embeddings are normalized

    def pick_best_context(self, message: str, history: List[dict]) -> tuple[Optional[str], float]:
        msg_norm = normalize_text(message)
        qvec = self.rag._get_query_embedding(msg_norm)
        best_text = None
        best_score = 0.0
        for h in reversed(history):
            if h.get("role") != "user":
                continue
            text = h.get("content", "")
            if not text:
                continue
            hvec = self.rag._get_query_embedding(text)
            sim = self._similarity(qvec, hvec)
            if sim > best_score:
                best_score = sim
                best_text = text
        return best_text, best_score

    def should_augment(self, top_score: float, second_score: float) -> bool:
        if top_score < MIN_RETRIEVAL_SCORE:
            return True
        if (top_score - second_score) < RETRIEVAL_CONFIDENCE_MARGIN:
            return True
        return False


# ─────────────────────────── Answer Guard ───────────────────────────
def _contains_any(text: str, phrases: List[str]) -> bool:
    t = text.lower()
    return any(p.lower() in t for p in phrases)


def guard_answer(answer: str, kb_context: str) -> str:
    """
    Remove any sentences that contain forbidden phrases unless those phrases
    appear in the retrieved KB chunk.
    """
    violations = []
    for phrases in FORBIDDEN_PHRASES.values():
        for phrase in phrases:
            if _contains_any(answer, [phrase]) and not _contains_any(kb_context, [phrase]):
                violations.append(phrase)

    if not violations:
        return answer

    # Split into sentences and drop those containing forbidden phrases
    sentences = re.split(r'(?<=[।!?])\s+', answer.strip())
    kept = []
    for s in sentences:
        if not s:
            continue
        if _contains_any(s, violations):
            continue
        kept.append(s)

    cleaned = " ".join(kept).strip()
    if cleaned:
        return cleaned
    return "দুঃখিত, এই বিষয়ে আমাদের কাছে তথ্য নেই।"


def _join_products(items: List[str]) -> str:
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} ও {items[1]}"
    return ", ".join(items[:-1]) + " ও " + items[-1]


def _clean_product_names(items: List[str], limit: int = 5) -> List[str]:
    out = []
    for n in items:
        clean = n.split(",")[0].strip()
        if clean and clean not in out:
            out.append(clean)
        if len(out) >= limit:
            break
    return out


def _intent_label_bn(intent: Optional[str]) -> str:
    mapping = {
        "rating": "রিভিউ",
        "price": "দাম",
        "warranty": "ওয়ারেন্টি",
        "delivery": "ডেলিভারি",
        "features": "বৈশিষ্ট্য",
        "discount": "ছাড়",
        "payment": "পেমেন্ট",
        "availability": "উপলব্ধতা",
        "popularity": "জনপ্রিয়তা",
    }
    return mapping.get(intent, "তথ্য")


def _resolve_product_name_from_token(rag: "RAGEngine", token: str) -> Optional[str]:
    candidates = rag.product_token_map.get(token, [])
    if not candidates:
        return None
    unique = sorted(set(candidates), key=lambda s: (len(s), s))
    return unique[0]


def _product_has_price_info(rag: "RAGEngine", product_name: str) -> bool:
    docs = rag.docs_for_product(product_name)
    return any(re.search(r"\d+\s*টাকা", d) for d in docs)


def _build_multi_product_answer(
    rag: "RAGEngine",
    terms: List[str],
    intent: Optional[str],
    msg_lower: str,
) -> Optional[str]:
    if not intent:
        return None
    parts: List[str] = []
    for term in terms:
        term_norm = normalize_text(term).lower()
        product_name = None
        if term in rag.product_token_map:
            product_name = _resolve_product_name_from_token(rag, term)
        elif term_norm in rag.product_names:
            product_name = term
        if product_name:
            facts = extract_product_facts(rag, product_name)
            if intent == "availability":
                parts.append(f"হ্যাঁ, আমাদের স্টোরে {term} পাওয়া যায়।")
            elif intent == "discount":
                disc = facts.get("discount_sentence", "")
                if disc:
                    if ("টাকা" in msg_lower) and not facts.get("has_money_amount"):
                        parts.append(f"{term} সম্পর্কে টাকার পরিমাণ উল্লেখ নেই, তবে {disc}।")
                    else:
                        parts.append(f"{term} সম্পর্কে {disc}।")
                else:
                    parts.append(f"দুঃখিত, {term} সম্পর্কে এই বিষয়ে আমাদের কাছে তথ্য নেই।")
            elif intent == "warranty":
                brands = sorted(facts.get("warranty_brands", []))
                if brands:
                    parts.append(f"{term} এর ওয়ারেন্টি আছে। ব্র্যান্ড: {', '.join(brands[:3])}।")
                else:
                    parts.append(f"দুঃখিত, {term} সম্পর্কে এই বিষয়ে আমাদের কাছে তথ্য নেই।")
            elif intent == "delivery":
                p = []
                if facts.get("fast_delivery"):
                    p.append("দ্রুত ডেলিভারি পাওয়া যায়")
                if facts.get("home_delivery"):
                    p.append("সারা বাংলাদেশে হোম ডেলিভারি সুবিধা আছে")
                if p:
                    parts.append(f"{term} সম্পর্কে " + "। ".join(p) + "।")
                else:
                    parts.append(f"দুঃখিত, {term} সম্পর্কে এই বিষয়ে আমাদের কাছে তথ্য নেই।")
            elif intent == "payment":
                methods = sorted(facts.get("payment_methods", []))
                if methods:
                    parts.append(f"{term} এর পেমেন্ট অপশন: {', '.join(methods)}।")
                else:
                    parts.append(f"দুঃখিত, {term} সম্পর্কে এই বিষয়ে আমাদের কাছে তথ্য নেই।")
            elif intent == "features":
                if facts.get("feature_sentence"):
                    parts.append(f"{term} সম্পর্কে {facts['feature_sentence']}।")
                else:
                    parts.append(f"দুঃখিত, {term} সম্পর্কে এই বিষয়ে আমাদের কাছে তথ্য নেই।")
            elif intent == "rating":
                if facts.get("rating"):
                    parts.append(f"{term} এর রিভিউ অনুযায়ী এটি {facts['rating']} স্টার রেটিং পেয়েছে।")
                else:
                    parts.append(f"দুঃখিত, {term} সম্পর্কে এই বিষয়ে আমাদের কাছে তথ্য নেই।")
            elif intent == "popularity":
                if facts.get("best_selling") or facts.get("popular"):
                    parts.append(f"{term} বাংলাদেশের বাজারে জনপ্রিয় এবং বহুল ব্যবহৃত।")
                else:
                    parts.append(f"দুঃখিত, {term} সম্পর্কে এই বিষয়ে আমাদের কাছে তথ্য নেই।")
            elif intent == "price":
                if product_name and _product_has_price_info(rag, product_name):
                    parts.append(f"দুঃখিত, {term} সম্পর্কে এই বিষয়ে আমাদের কাছে তথ্য নেই।")
                else:
                    parts.append(f"দুঃখিত, {term} সম্পর্কে এই বিষয়ে আমাদের কাছে তথ্য নেই।")
            else:
                return None
        else:
            parts.append(f"দুঃখিত, {term} আমাদের স্টোরে পাওয়া যায় না।")
    return " ".join(parts) if parts else None


# ─────────────────────────── Auth Helpers ───────────────────────────
def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)

def hash_password(plain: str) -> str:
    return pwd_context.hash(plain)

def create_access_token(data: dict, expires_delta: timedelta = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
    to_encode["exp"] = expire
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def get_current_user(token: str = Depends(oauth2_scheme)):
    cred_exc = HTTPException(status_code=401, detail="Invalid credentials",
                             headers={"WWW-Authenticate": "Bearer"})
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if not username:
            raise cred_exc
    except JWTError:
        raise cred_exc
    conn = get_db()
    user = conn.execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()
    conn.close()
    if not user:
        raise cred_exc
    return dict(user)


# ─────────────────────────── Pydantic Models ───────────────────────────
class RegisterRequest(BaseModel):
    username: str
    email: str
    password: str

class ChatRequest(BaseModel):
    message: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str
    username: str


# ─────────────────────────── App Startup ───────────────────────────
app = FastAPI(title="Bengali RAG Chatbot")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

init_db()
rag = RAGEngine(KB_PATH)
ctx_manager = ContextManager(rag)
semantic_ctx = SemanticContextResolver(rag)

app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")), name="static")


# ─────────────────────────── Auth Routes ───────────────────────────
@app.get("/")
def root():
    return FileResponse(os.path.join(os.path.dirname(__file__), "static", "login.html"))

@app.get("/chat")
def chat_page():
    return FileResponse(os.path.join(os.path.dirname(__file__), "static", "chat.html"))

@app.get("/register-page")
def register_page():
    return FileResponse(os.path.join(os.path.dirname(__file__), "static", "register.html"))

@app.post("/api/register", status_code=201)
def register(req: RegisterRequest):
    if len(req.username) < 3:
        raise HTTPException(400, "Username must be at least 3 characters")
    if len(req.password) < 6:
        raise HTTPException(400, "Password must be at least 6 characters")
    conn = get_db()
    try:
        conn.execute(
            "INSERT INTO users (username, email, hashed_password) VALUES (?,?,?)",
            (req.username.strip(), req.email.strip(), hash_password(req.password))
        )
        conn.commit()
    except sqlite3.IntegrityError as e:
        conn.close()
        if "username" in str(e):
            raise HTTPException(409, "Username already taken")
        raise HTTPException(409, "Email already registered")
    conn.close()
    return {"message": "Registration successful"}

@app.post("/api/token", response_model=TokenResponse)
def login(form: OAuth2PasswordRequestForm = Depends()):
    conn = get_db()
    user = conn.execute("SELECT * FROM users WHERE username=?", (form.username,)).fetchone()
    conn.close()
    if not user or not verify_password(form.password, user["hashed_password"]):
        raise HTTPException(401, "Incorrect username or password")
    token = create_access_token({"sub": user["username"]},
                                timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    return {"access_token": token, "token_type": "bearer", "username": user["username"]}

@app.get("/api/me")
def me(current_user=Depends(get_current_user)):
    return {"username": current_user["username"], "email": current_user["email"]}


# ─────────────────────────── Chat Route ───────────────────────────
@app.post("/api/chat")
def chat(req: ChatRequest, current_user=Depends(get_current_user)):
    if not req.message.strip():
        raise HTTPException(400, "Empty message")

    user_id = current_user["id"]
    conn = get_db()

    # Load last CONTEXT_WINDOW turns
    rows = conn.execute(
        "SELECT role, content FROM chat_history WHERE user_id=? ORDER BY id DESC LIMIT ?",
        (user_id, CONTEXT_WINDOW * 2)
    ).fetchall()
    history = [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]

    msg_norm = normalize_text(req.message)
    msg_lower = msg_norm.lower()

    # Detect product mentions in the current message (without history)
    msg_matched_names = rag.matched_product_names(msg_norm)
    msg_product_token_hit = rag.has_product_token(msg_norm)
    product_in_msg = len(msg_matched_names) > 0 or msg_product_token_hit
    msg_has_own_context = ctx_manager.has_own_context(msg_norm)
    multi_product = ctx_manager.is_multi_product(msg_norm)
    multi_candidates = ctx_manager.extract_candidate_products(msg_norm) if multi_product else []
    recent_known, recent_candidates, recent_has_multi = ([], [], False)
    if not msg_has_own_context:
        recent_known, recent_candidates, recent_has_multi = ctx_manager.extract_recent_products_with_candidates(history)
    multi_targets = []
    if msg_has_own_context and multi_product:
        if len(msg_matched_names) >= 2:
            multi_targets = [n.split(",")[0].strip() for n in msg_matched_names]
        else:
            multi_targets = multi_candidates
    elif (not msg_has_own_context) and recent_has_multi:
        if len(recent_known) >= 2:
            multi_targets = recent_known
        elif len(recent_candidates) >= 2:
            multi_targets = recent_candidates

    # If the message has a known product token but ALSO has an unknown brand/qualifier
    # (e.g. "নাইকি জুতা আছে?") → treat it as "not in store" so we don't answer about
    # the generic product and don't pollute last_product with a wrong match.
    has_unknown_brand = (not multi_product) and product_in_msg and rag.has_unknown_product_qualifiers(msg_norm)

    # Intent detection for follow-ups
    is_price_query = any(k in msg_lower for k in ["দাম", "price", "কত টাকা", "টাকা কত"])
    is_discount_query = any(k in msg_lower for k in ["ছাড়", "ছাড়", "discount", "অফার", "বিশেষ মূল্য"])
    is_availability_query = any(k in msg_lower for k in ["বিক্রি", "কেনা", "পাওয়া যায়", "পাওয়া যায়", "পাওয়া যাবে", "পাওয়া যাবে", "আছে", "sell", "buy", "available"])
    is_feature_query = any(k in msg_lower for k in ["ফিচার", "বৈশিষ্ট্য", "features"])
    is_warranty_query = any(k in msg_lower for k in ["ওয়ারেন্টি", "ওয়ারেন্টি", "warranty"])
    is_delivery_query = any(k in msg_lower for k in ["ডেলিভারি", "ডেলিভারী", "delivery"])
    is_payment_query = any(k in msg_lower for k in ["পেমেন্ট", "payment", "বিকাশ", "নগদ", "কার্ড"])
    is_rating_query = any(k in msg_lower for k in ["রেটিং", "রিভিউ", "স্টার"])
    is_popularity_query = any(k in msg_lower for k in ["জনপ্রিয়", "সবচেয়ে বেশি", "বিক্রিত", "বহুল ব্যবহৃত"])

    followup_intent = any([
        is_price_query,
        is_discount_query,
        is_availability_query,
        is_feature_query,
        is_warranty_query,
        is_delivery_query,
        is_payment_query,
        is_rating_query,
        is_popularity_query,
    ])

    # Domain classification (store vs general chat)
    query_domain = rag.query_domain(msg_norm)
    semantic_hit_domain, _, _ = rag.semantic_product_match(msg_norm)
    if semantic_hit_domain or product_in_msg:
        query_domain = "store"

    # Primary intent (single)
    primary_intent = None
    if is_discount_query:
        primary_intent = "discount"
    elif is_price_query:
        primary_intent = "price"
    elif is_warranty_query:
        primary_intent = "warranty"
    elif is_delivery_query:
        primary_intent = "delivery"
    elif is_payment_query:
        primary_intent = "payment"
    elif is_feature_query:
        primary_intent = "features"
    elif is_rating_query:
        primary_intent = "rating"
    elif is_popularity_query:
        primary_intent = "popularity"
    elif is_availability_query:
        primary_intent = "availability"

    has_followup_cue = any(k in msg_lower for k in ["আর", "আরও", "এছাড়া", "এছাড়া", "আগেরটা", "ওটা", "এটা", "এইটা"])

    # Initial RAG query (semantic context resolver may adjust after first retrieval)
    rag_query = req.message
    rag_query_norm = normalize_text(rag_query)

    # Use normalized query to detect product mentions
    matched_names = rag.matched_product_names(rag_query_norm)
    product_in_query = len(matched_names) > 0
    product_token_hit = rag.has_product_token(rag_query_norm)
    semantic_product_hit = False
    semantic_product_score = 0.0
    semantic_product_name = None
    if query_domain == "store":
        semantic_product_hit, semantic_product_score, semantic_product_name = rag.semantic_product_match(rag_query_norm)
        if semantic_product_hit:
            product_in_query = True
            if (not matched_names) and semantic_product_score >= PRODUCT_NAME_BIND_THRESHOLD and semantic_product_name:
                matched_names = [semantic_product_name]
                product_token_hit = True

    # Decide which product(s) to store as last_product (even if unknown)
    state_products: List[str] = []
    state_known = False
    if msg_has_own_context:
        if multi_product:
            if multi_candidates:
                state_products = multi_candidates
                state_known = bool(msg_matched_names)
            elif msg_matched_names:
                state_products = [n.split(",")[0].strip() for n in msg_matched_names]
                state_known = True
        else:
            if msg_matched_names:
                if len(msg_matched_names) == 1:
                    state_products = [msg_matched_names[0].split(",")[0].strip()]
                    state_known = True
                else:
                    # Ambiguous match (e.g., many noodle variants) -> store the user's category token
                    candidates = ctx_manager.extract_candidate_products(msg_norm)
                    if candidates:
                        state_products = [" ".join(candidates)]
                        state_known = True
            if not state_products:
                candidates = ctx_manager.extract_candidate_products(msg_norm)
                if candidates:
                    state_products = [" ".join(candidates)]
                    state_known = False
    if not state_products and semantic_product_hit and semantic_product_name:
        state_products = [semantic_product_name]
        state_known = True

    if has_unknown_brand:
        state_known = False

    # Context memory: fallback to last product from state (used only if semantic context not applied)
    last_product, last_intent, last_product_known = get_user_state(user_id)

    if not primary_intent and last_intent and (has_followup_cue or product_in_query):
        primary_intent = last_intent

    # General-chat fallback (non-product queries)
    if (query_domain == "general") and (not followup_intent):
        if not groq_client:
            answer = "আমি সাধারণ কথোপকথনে সাহায্য করতে পারি, বলুন কী জানতে চান?"
            conn.execute("INSERT INTO chat_history (user_id, role, content) VALUES (?,?,?)",
                         (user_id, "user", req.message))
            conn.execute("INSERT INTO chat_history (user_id, role, content) VALUES (?,?,?)",
                         (user_id, "assistant", answer))
            conn.commit()
            conn.close()
            return {
                "answer": answer,
                "retrieval_ms": 0.0,
                "generation_ms": 0.0,
                "kb_hits": 0,
                "debug": {
                    "rag_query": "",
                    "rag_query_norm": "",
                    "top_score": 0.0,
                    "second_score": 0.0,
                    "has_kb_match": False,
                    "matched_names": [],
                    "last_product": None,
                    "last_intent": None,
                    "followup_intent": False,
                    "primary_intent": None,
                    "product_token_hit": False,
                    "discount_sentence": "",
                    "product_in_query": False,
                    "is_price_query": False,
                    "is_discount_query": False,
                    "is_availability_query": False,
                    "is_feature_query": False,
                    "is_warranty_query": False,
                    "is_delivery_query": False,
                    "is_payment_query": False,
                    "has_price_info": False,
                    "decision": "general_chat_no_llm",
                    "raw_answer": "",
                    "semantic_context_used": False,
                    "semantic_context_score": 0.0,
                    "semantic_context_text": None,
                    "query_domain": query_domain,
                }
            }

        system_prompt = (
            "You are a friendly Bengali conversational assistant. "
            "Respond naturally and briefly. If the user asks about store products, prices, "
            "availability, warranty, delivery, payment or discounts, say you can only answer "
            "using the store's knowledge base."
        )
        t_gen = time.time()
        try:
            response = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": req.message},
                ],
                max_tokens=128,
                temperature=0.4,
            )
            answer = response.choices[0].message.content.strip()
        except Exception as e:
            raise HTTPException(500, f"LLM error: {str(e)}")
        generation_ms = (time.time() - t_gen) * 1000
        conn.execute("INSERT INTO chat_history (user_id, role, content) VALUES (?,?,?)",
                     (user_id, "user", req.message))
        conn.execute("INSERT INTO chat_history (user_id, role, content) VALUES (?,?,?)",
                     (user_id, "assistant", answer))
        conn.commit()
        conn.close()
        return {
            "answer": answer,
            "retrieval_ms": 0.0,
            "generation_ms": round(generation_ms, 2),
            "kb_hits": 0,
            "debug": {
                "rag_query": "",
                "rag_query_norm": "",
                "top_score": 0.0,
                "second_score": 0.0,
                "has_kb_match": False,
                "matched_names": [],
                "last_product": None,
                "last_intent": None,
                "followup_intent": False,
                "primary_intent": None,
                "product_token_hit": False,
                "discount_sentence": "",
                "product_in_query": False,
                "is_price_query": False,
                "is_discount_query": False,
                "is_availability_query": False,
                "is_feature_query": False,
                "is_warranty_query": False,
                "is_delivery_query": False,
                "is_payment_query": False,
                "has_price_info": False,
                "decision": "general_chat_llm",
                "raw_answer": "",
                "semantic_context_used": False,
                "semantic_context_score": 0.0,
                "semantic_context_text": None,
                "query_domain": query_domain,
            }
        }

    # If followup refers to an unknown product, respond "not available"
    if (
        not msg_has_own_context
        and followup_intent
        and last_product
        and not last_product_known
    ):
        answer = "দুঃখিত, এই পণ্যটি আমাদের স্টোরে পাওয়া যায় না।"
        decision = "followup_unknown_product"
        generation_ms = 0.0
        conn.execute("INSERT INTO chat_history (user_id, role, content) VALUES (?,?,?)",
                     (user_id, "user", req.message))
        conn.execute("INSERT INTO chat_history (user_id, role, content) VALUES (?,?,?)",
                     (user_id, "assistant", answer))
        conn.commit()
        conn.close()
        return {
            "answer": answer,
            "retrieval_ms": 0.0,
            "generation_ms": 0.0,
            "kb_hits": 0,
            "debug": {
                "rag_query": "",
                "rag_query_norm": "",
                "top_score": 0.0,
                "second_score": 0.0,
                "has_kb_match": False,
                "matched_names": [],
                "semantic_product_score": round(semantic_product_score, 4),
                "state_products": state_products,
                "last_product": last_product,
                "last_product_known": last_product_known,
                "last_intent": last_intent,
                "followup_intent": followup_intent,
                "primary_intent": primary_intent,
                "product_token_hit": False,
                "discount_sentence": "",
                "product_in_query": False,
                "is_price_query": is_price_query,
                "is_discount_query": is_discount_query,
                "is_availability_query": is_availability_query,
                "is_feature_query": is_feature_query,
                "is_warranty_query": is_warranty_query,
                "is_delivery_query": is_delivery_query,
                "is_payment_query": is_payment_query,
                "has_price_info": False,
                "decision": decision,
                "raw_answer": "",
                "semantic_context_used": False,
                "semantic_context_score": 0.0,
                "semantic_context_text": None,
                "query_domain": query_domain,
            }
        }

    # Retrieve from KB (first pass)
    t_ret = time.time()
    kb_chunks, top_score, second_score = rag.retrieve(rag_query_norm)
    retrieval_ms = (time.time() - t_ret) * 1000

    # Semantic context fallback (non-rule-based)
    semantic_context_used = False
    semantic_context_text = None
    semantic_context_score = 0.0
    if (not msg_has_own_context) and (not product_in_query) and (query_domain == "store") and semantic_ctx.should_augment(top_score, second_score):
        best_text, best_sim = semantic_ctx.pick_best_context(req.message, history)
        best_text_norm = normalize_text(best_text) if best_text else ""
        msg_norm_cmp = normalize_text(req.message)
        best_has_product = bool(rag.matched_product_names(best_text_norm)) or rag.has_product_token(best_text_norm) if best_text else False
        if (
            best_text
            and best_sim >= SEMANTIC_CONTEXT_MIN_SIM
            and best_text_norm != msg_norm_cmp
            and best_has_product
        ):
            semantic_context_used = True
            semantic_context_text = best_text
            semantic_context_score = best_sim
            rag_query = f"{best_text} {req.message}"
            rag_query_norm = normalize_text(rag_query)
            matched_names = rag.matched_product_names(rag_query_norm)
            product_in_query = len(matched_names) > 0
            product_token_hit = rag.has_product_token(rag_query_norm)
            t_ret = time.time()
            kb_chunks, top_score, second_score = rag.retrieve(rag_query_norm)
            retrieval_ms = (time.time() - t_ret) * 1000

    # Fallback to last_product only if semantic context was not used
    if (
        not semantic_context_used
        and not msg_has_own_context
        and last_product
        and last_product_known
        and (followup_intent or has_followup_cue or ctx_manager._is_question_like(req.message))
        and rag_query == req.message
    ):
        rag_query = f"{last_product} {rag_query}"
        rag_query_norm = normalize_text(rag_query)
        matched_names = rag.matched_product_names(rag_query_norm)
        product_in_query = len(matched_names) > 0
        product_token_hit = rag.has_product_token(rag_query_norm)
        t_ret = time.time()
        kb_chunks, top_score, second_score = rag.retrieve(rag_query_norm)
        retrieval_ms = (time.time() - t_ret) * 1000

    has_kb_match = len(kb_chunks) > 0 and top_score >= MIN_RETRIEVAL_SCORE

    # Fallback: direct product-name lookup before giving up.
    if not has_kb_match and product_in_query:
        fallback_chunks = rag.retrieve_by_product_name(rag_query_norm)
        if fallback_chunks:
            kb_chunks = fallback_chunks
            has_kb_match = True

    has_price_info = any(("টাকা" in c) or re.search(r"\d", c) for c in kb_chunks)

    facts = extract_product_facts(rag, matched_names[0]) if matched_names else {}
    discount_sentence = facts.get("discount_sentence", "")

    decision = ""
    raw_answer = ""
    # Brand mismatch: user asked for a specific brand not in our KB
    # (e.g. "নাইকি জুতা আছে?" — নাইকি is unknown, জুতা is a generic KB product).
    # Return "not available" immediately and don't store anything as last_product.
    if multi_targets and primary_intent:
        multi_answer = _build_multi_product_answer(rag, multi_targets, primary_intent, msg_lower)
        if multi_answer:
            answer = multi_answer
            decision = "multi_product_intent"
            generation_ms = 0.0
            # Update last_product with known tokens only
            known_multi = []
            for t in multi_targets:
                t_norm = normalize_text(t).lower()
                if t in rag.product_token_map or t_norm in rag.product_names:
                    known_multi.append(t)
            matched_names = known_multi
        else:
            answer = "দুঃখিত, এই বিষয়ে আমাদের কাছে তথ্য নেই।"
            decision = "multi_product_no_info"
            generation_ms = 0.0
            matched_names = []
    elif has_unknown_brand:
        answer = "দুঃখিত, এই পণ্যটি আমাদের স্টোরে পাওয়া যায় না।"
        decision = "unknown_brand"
        generation_ms = 0.0
        matched_names = []  # ensure last_product is not updated
    elif primary_intent == "availability" and (not matched_names) and (not product_token_hit) and (not semantic_product_hit):
        answer = "দুঃখিত, এই পণ্যটি আমাদের স্টোরে পাওয়া যায় না।"
        decision = "availability_no_match"
        generation_ms = 0.0
    elif matched_names and primary_intent:
        if len(matched_names) > 1 and primary_intent in {"rating", "price", "warranty", "delivery", "features", "discount"}:
            variants = _clean_product_names(matched_names, limit=5)
            variants_str = _join_products(variants)
            answer = f"একাধিক ভ্যারিয়েন্ট আছে: {variants_str}। আপনি কোনটির {_intent_label_bn(primary_intent)} জানতে চান?"
            decision = "variant_clarify"
            generation_ms = 0.0
        else:
            if primary_intent == "availability":
                if len(matched_names) > 1:
                    variants = _clean_product_names(matched_names, limit=5)
                    variants_str = _join_products(variants)
                    answer = f"হ্যাঁ, আমাদের স্টোরে কয়েকটি ভ্যারিয়েন্ট পাওয়া যায়: {variants_str}। আপনি কোনটা চান?"
                else:
                    answer = f"হ্যাঁ, আমাদের স্টোরে {matched_names[0]} পাওয়া যায়।"
            elif primary_intent == "discount":
                if discount_sentence:
                    if "টাকা" in msg_lower and not facts.get("has_money_amount"):
                        answer = f"টাকার পরিমাণ উল্লেখ নেই, তবে {discount_sentence}।"
                    else:
                        answer = f"{discount_sentence}।"
                else:
                    answer = "দুঃখিত, এই বিষয়ে আমাদের কাছে তথ্য নেই।"
            elif primary_intent == "warranty":
                brands = sorted(facts.get("warranty_brands", []))
                if brands:
                    answer = f"এই পণ্যে ওয়ারেন্টি আছে। ব্র্যান্ড: {', '.join(brands[:3])}।"
                else:
                    answer = "দুঃখিত, এই বিষয়ে আমাদের কাছে তথ্য নেই।"
            elif primary_intent == "delivery":
                parts = []
                if facts.get("fast_delivery"):
                    parts.append("দ্রুত ডেলিভারি পাওয়া যায়")
                if facts.get("home_delivery"):
                    parts.append("সারা বাংলাদেশে হোম ডেলিভারি সুবিধা আছে")
                if parts:
                    answer = "। ".join(parts) + "।"
                else:
                    answer = "দুঃখিত, এই বিষয়ে আমাদের কাছে তথ্য নেই।"
            elif primary_intent == "payment":
                methods = sorted(facts.get("payment_methods", []))
                if methods:
                    answer = f"পেমেন্ট অপশন: {', '.join(methods)}।"
                else:
                    answer = "দুঃখিত, এই বিষয়ে আমাদের কাছে তথ্য নেই।"
            elif primary_intent == "features":
                if facts.get("feature_sentence"):
                    answer = f"{facts['feature_sentence']}।"
                else:
                    answer = "দুঃখিত, এই বিষয়ে আমাদের কাছে তথ্য নেই।"
            elif primary_intent == "rating":
                if facts.get("rating"):
                    answer = f"গ্রাহকদের রিভিউ অনুযায়ী এটি {facts['rating']} স্টার রেটিং পেয়েছে।"
                else:
                    answer = "দুঃখিত, এই বিষয়ে আমাদের কাছে তথ্য নেই।"
            elif primary_intent == "popularity":
                if facts.get("best_selling") or facts.get("popular"):
                    answer = "এই পণ্যটি বাংলাদেশের বাজারে জনপ্রিয় এবং বহুল ব্যবহৃত।"
                else:
                    answer = "দুঃখিত, এই বিষয়ে আমাদের কাছে তথ্য নেই।"
            elif primary_intent == "price":
                if has_price_info:
                    answer = "দুঃখিত, এই বিষয়ে আমাদের কাছে তথ্য নেই।"
                else:
                    answer = "দুঃখিত, এই বিষয়ে আমাদের কাছে তথ্য নেই।"
            else:
                answer = "দুঃখিত, এই বিষয়ে আমাদের কাছে তথ্য নেই।"
            generation_ms = 0.0
            decision = f"{primary_intent}_intent"
    elif not has_kb_match:
        if is_price_query or is_availability_query or product_in_query:
            answer = "দুঃখিত, এই পণ্যটি আমাদের স্টোরে পাওয়া যায় না।"
            decision = "fallback_not_available"
        else:
            answer = "দুঃখিত, এই বিষয়ে আমাদের কাছে তথ্য নেই।"
            decision = "fallback_no_info"
        generation_ms = 0.0
    elif is_price_query and not has_price_info:
        answer = "দুঃখিত, এই বিষয়ে আমাদের কাছে তথ্য নেই।"
        decision = "price_no_info"
        generation_ms = 0.0
    else:
        # Build system prompt
        # Limit KB context to avoid oversized prompt
        max_kb_chars = 3500
        kb_context = "\n\n".join(kb_chunks)
        kb_context = truncate_text(kb_context, max_kb_chars)

        # Tell the LLM which products matched so it won't hallucinate other product names.
        if matched_names:
            names_str = ", ".join(f'"{n}"' for n in matched_names[:3])
            product_hint = (
                f"\n\n⚠️ CONFIRMED CATALOG MATCH: The following product(s) from the Knowledge Base "
                f"match the user's query: {names_str}. "
                "You MUST confirm these products are available in our store. Do NOT say they are unavailable."
            )
        else:
            product_hint = ""

        system_prompt = f"""You are a helpful Bengali e-commerce assistant. Your ONLY source of truth is the Knowledge Base provided below. Never use outside knowledge or assumptions.

STRICT RULES (violating any rule is not allowed):
1. Always reply in Bengali.
2. Use ONLY information from the Knowledge Base — never invent, guess, or add anything from your own training.
3. A product is "available" if and only if its name appears in the Knowledge Base. Partial name matches count — e.g., if the KB contains "এনালগ ও ডিজিটাল ঘড়ি" and the user asks about "ডিজিটাল ঘড়ি", the product IS available.
4. If a product is NOT in the Knowledge Base → say exactly: "দুঃখিত, এই পণ্যটি আমাদের স্টোরে পাওয়া যায় না।"
5. If information is NOT in the Knowledge Base → say exactly: "দুঃখিত, এই বিষয়ে আমাদের কাছে তথ্য নেই।"
6. NEVER mention any product name that is not present in the retrieved Knowledge Base chunks below.
7. NEVER mention discounts, special prices, or gift wrapping unless the Knowledge Base explicitly states it.
8. Keep answers short, clear, and helpful.{product_hint}

--- Knowledge Base (retrieved context) ---
{kb_context}
--- End of Knowledge Base ---"""

        # Build messages for LLM
        # When a catalog product is matched, annotate the user message so the LLM
        # cannot mistake a partial name ("ডিজিটাল ঘড়ি") for an unknown product.
        if matched_names:
            catalog_note = (
                f"[CATALOG NOTE: The user is asking about a product. "
                f"Our store sells: {', '.join(matched_names[:3])}. "
                "This product IS available. Confirm availability and give details from the Knowledge Base.]"
            )
            user_msg_for_llm = f"{catalog_note}\n\n{req.message}"
        else:
            user_msg_for_llm = req.message

        messages = [{"role": "system", "content": system_prompt}]
        # Keep history short to avoid prompt overflow
        for h in history[-CONTEXT_WINDOW:]:
            messages.append({
                "role": h["role"],
                "content": truncate_text(h["content"], 400)
            })
        messages.append({"role": "user", "content": user_msg_for_llm})

        # Call Groq LLM
        if not groq_client:
            raise HTTPException(500, "GROQ_API_KEY not configured")

        t_gen = time.time()
        try:
            response = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages,
                max_tokens=256,
                temperature=0.3,
            )
            raw_answer = response.choices[0].message.content.strip()
            answer = guard_answer(raw_answer, kb_context)
            decision = "llm_answer"
        except Exception as e:
            raise HTTPException(500, f"LLM error: {str(e)}")
        generation_ms = (time.time() - t_gen) * 1000

    # Save to history
    if state_products or primary_intent:
        last_product_value = _join_products(state_products) if state_products else None
        set_user_state(
            user_id,
            product_name=last_product_value,
            intent=primary_intent,
            product_known=state_known if state_products else None
        )
    conn.execute("INSERT INTO chat_history (user_id, role, content) VALUES (?,?,?)",
                 (user_id, "user", req.message))
    conn.execute("INSERT INTO chat_history (user_id, role, content) VALUES (?,?,?)",
                 (user_id, "assistant", answer))
    conn.commit()
    conn.close()

    # Debug logging
    print(
        "[CHAT]",
        f"q='{req.message}'",
        f"rag_q='{rag_query}'",
        f"rag_q_norm='{rag_query_norm}'",
        f"top_score={top_score:.4f}",
        f"second_score={second_score:.4f}",
        f"hits={len(kb_chunks)}",
        f"has_kb_match={has_kb_match}",
        f"matched_names={matched_names}",
        f"semantic_product_score={semantic_product_score:.4f}",
        f"state_products={state_products}",
        f"last_product={last_product}",
        f"last_product_known={last_product_known}",
        f"last_intent={last_intent}",
        f"followup_intent={followup_intent}",
        f"primary_intent={primary_intent}",
        f"product_token_hit={product_token_hit}",
        f"discount_sentence={discount_sentence}",
        f"decision={decision}",
        f"semantic_ctx_used={semantic_context_used}",
        f"semantic_ctx_score={semantic_context_score:.4f}",
        f"query_domain={query_domain}",
        f"retrieval_ms={retrieval_ms:.2f}",
        f"generation_ms={generation_ms:.2f}",
    )

    return {
        "answer": answer,
        "retrieval_ms": round(retrieval_ms, 2),
        "generation_ms": round(generation_ms, 2),
        "kb_hits": len(kb_chunks),
        "debug": {
            "rag_query": rag_query,
            "rag_query_norm": rag_query_norm,
            "top_score": round(top_score, 4),
            "second_score": round(second_score, 4),
            "has_kb_match": has_kb_match,
            "matched_names": matched_names,
            "semantic_product_score": round(semantic_product_score, 4),
            "state_products": state_products,
            "last_product": last_product,
            "last_product_known": last_product_known,
            "last_intent": last_intent,
            "followup_intent": followup_intent,
            "semantic_context_used": semantic_context_used,
            "semantic_context_score": round(semantic_context_score, 4),
            "semantic_context_text": semantic_context_text,
            "query_domain": query_domain,
            "primary_intent": primary_intent,
            "product_token_hit": product_token_hit,
            "discount_sentence": discount_sentence,
            "product_in_query": product_in_query,
            "is_price_query": is_price_query,
            "is_discount_query": is_discount_query,
            "is_availability_query": is_availability_query,
            "is_feature_query": is_feature_query,
            "is_warranty_query": is_warranty_query,
            "is_delivery_query": is_delivery_query,
            "is_payment_query": is_payment_query,
            "has_price_info": has_price_info,
            "decision": decision,
            "raw_answer": raw_answer,
        }
    }

@app.get("/api/history")
def get_history(current_user=Depends(get_current_user)):
    conn = get_db()
    rows = conn.execute(
        "SELECT role, content, created_at FROM chat_history WHERE user_id=? ORDER BY id DESC LIMIT 50",
        (current_user["id"],)
    ).fetchall()
    conn.close()
    return [{"role": r["role"], "content": r["content"], "time": r["created_at"]}
            for r in reversed(rows)]

@app.delete("/api/history")
def clear_history(current_user=Depends(get_current_user)):
    conn = get_db()
    conn.execute("DELETE FROM chat_history WHERE user_id=?", (current_user["id"],))
    conn.execute("DELETE FROM user_state WHERE user_id=?", (current_user["id"],))
    conn.commit()
    conn.close()
    return {"message": "History cleared"}


# ─────────────────────────── Run ───────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main_semantic:app", host="0.0.0.0", port=8000, reload=False)
