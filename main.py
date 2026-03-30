import os
import re
import time
import math
import json
import sqlite3
import hashlib
import secrets
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from typing import Optional, List

import numpy as np
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
CONTEXT_WINDOW = 5  # last N turns kept for context
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
    """)
    conn.commit()
    conn.close()

# ─────────────────────────── TF-IDF RAG Engine ───────────────────────────
class RAGEngine:
    def __init__(self, kb_path: str):
        self.documents: List[str] = []
        self.tfidf_matrix: Optional[np.ndarray] = None
        self.vocab: dict = {}
        self.idf: Optional[np.ndarray] = None
        self.product_names: set[str] = set()
        self._load_and_index(kb_path)

    def _tokenize(self, text: str) -> List[str]:
        # Handle Bengali + English tokens
        text = text.lower()
        tokens = re.findall(r'[\u0980-\u09FF]+|[a-zA-Z0-9]+', text)
        return tokens

    def _load_and_index(self, path: str):
        t0 = time.time()
        with open(path, encoding="utf-8") as f:
            raw = f.read()

        # Split by blank lines; each paragraph = one document
        paras = [p.strip() for p in re.split(r'\n\s*\n', raw) if p.strip()]
        self.documents = paras
        n = len(paras)
        # Extract product names from first sentence of each paragraph
        for p in paras:
            first_sentence = p.split("।", 1)[0].strip()
            if first_sentence:
                self.product_names.add(first_sentence.lower())

        # Build vocabulary
        doc_tokens = [self._tokenize(d) for d in paras]
        all_tokens = set(t for tokens in doc_tokens for t in tokens)
        self.vocab = {t: i for i, t in enumerate(sorted(all_tokens))}
        V = len(self.vocab)

        # TF matrix (n x V)
        tf = np.zeros((n, V), dtype=np.float32)
        for i, tokens in enumerate(doc_tokens):
            cnt = Counter(tokens)
            total = len(tokens) or 1
            for tok, c in cnt.items():
                if tok in self.vocab:
                    tf[i, self.vocab[tok]] = c / total

        # IDF
        df = np.count_nonzero(tf, axis=0).astype(np.float32)
        self.idf = np.log((n + 1) / (df + 1)) + 1.0

        # TF-IDF
        tfidf = tf * self.idf
        # L2 normalise rows
        norms = np.linalg.norm(tfidf, axis=1, keepdims=True)
        norms[norms == 0] = 1
        self.tfidf_matrix = tfidf / norms

        print(f"[RAG] Indexed {n} docs, vocab={V}, took {(time.time()-t0)*1000:.1f}ms")

    def _query_vector(self, query: str) -> np.ndarray:
        tokens = self._tokenize(query)
        cnt = Counter(tokens)
        total = len(tokens) or 1
        V = len(self.vocab)
        vec = np.zeros(V, dtype=np.float32)
        for tok, c in cnt.items():
            if tok in self.vocab:
                vec[self.vocab[tok]] = (c / total) * self.idf[self.vocab[tok]]
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    def retrieve(self, query: str, top_k: int = TOP_K) -> List[str]:
        t0 = time.time()
        qvec = self._query_vector(query)
        scores = self.tfidf_matrix @ qvec  # cosine similarity
        top_idx = np.argsort(scores)[::-1][:top_k]
        results = [self.documents[i] for i in top_idx if scores[i] > 0.01]
        elapsed = (time.time() - t0) * 1000
        print(f"[RAG] Retrieval took {elapsed:.2f}ms, top_score={scores[top_idx[0]]:.4f}")
        return results

    def has_product_mention(self, message: str) -> bool:
        msg = message.lower()
        return any(name in msg for name in self.product_names)


# ─────────────────────────── Context Manager ───────────────────────────
class ContextManager:
    """
    Detects if the user's message contains its own context.
    If not, injects recent conversation history into the RAG query.
    """

    @staticmethod
    def has_context(message: str) -> bool:
        """Heuristic: message is self-contained if it's > 15 chars with a noun"""
        # Basic check: message mentions a product-like keyword
        indicators = ['কি', 'কী', 'কোথায়', 'কেন', 'কিভাবে', 'দাম', 'পণ্য',
                      'আছে', 'কত', 'what', 'how', 'price', 'product', 'buy']
        msg_lower = message.lower()
        return any(ind in msg_lower for ind in indicators) and len(message) > 8

    @staticmethod
    def build_rag_query(message: str, history: List[dict]) -> str:
        """Augment sparse queries with recent context."""
        if len(message.strip()) < 10 and history:
            # Very short follow-up — prepend last user message for context
            last_user_msgs = [h['content'] for h in history if h['role'] == 'user'][-2:]
            context_str = ' '.join(last_user_msgs)
            return f"{context_str} {message}"
        return message


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
ctx_manager = ContextManager()

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

    # Build RAG query with context augmentation
    rag_query = ctx_manager.build_rag_query(req.message, history)

    # Retrieve from KB
    t_ret = time.time()
    kb_chunks = rag.retrieve(rag_query)
    retrieval_ms = (time.time() - t_ret) * 1000

    msg_lower = req.message.lower()
    # Use context-augmented query to detect the product for follow-up questions
    product_in_query = rag.has_product_mention(rag_query)
    is_price_query = any(k in msg_lower for k in ["দাম", "price", "কত টাকা", "টাকা কত"])
    is_availability_query = any(k in msg_lower for k in ["বিক্রি", "কেনা", "পাওয়া যায়", "পাওয়া যায়", "sell", "buy", "available"])
    has_price_info = any(("টাকা" in c) or re.search(r"\d", c) for c in kb_chunks)

    if is_price_query and not product_in_query:
        answer = "দুঃখিত, এই বিষয়ে আমাদের কাছে তথ্য নেই।"
        generation_ms = 0.0
    elif is_availability_query and not product_in_query:
        answer = "দুঃখিত, এই পণ্যটি আমাদের স্টোরে পাওয়া যায় না।"
        generation_ms = 0.0
    elif not kb_chunks:
        answer = "দুঃখিত, এই পণ্যটি আমাদের স্টোরে পাওয়া যায় না।" if not product_in_query \
            else "দুঃখিত, এই বিষয়ে আমাদের কাছে তথ্য নেই।"
        generation_ms = 0.0
    elif is_price_query and not has_price_info:
        answer = "দুঃখিত, এই বিষয়ে আমাদের কাছে তথ্য নেই।"
        generation_ms = 0.0
    else:
        # Build system prompt
        kb_context = "\n\n".join(kb_chunks)
        system_prompt = f"""আপনি একটি বাংলাদেশি ই-কমার্স প্ল্যাটফর্মের সহায়ক চ্যাটবট। আপনি শুধুমাত্র নিচের জ্ঞান ভিত্তি (Knowledge Base) ব্যবহার করে উত্তর দেবেন।

নিয়মাবলী:
১. সর্বদা বাংলায় উত্তর দিন।
২. শুধুমাত্র Knowledge Base-এ থাকা তথ্য ব্যবহার করুন।
৩. যদি কোনো পণ্য Knowledge Base-এ না থাকে, বলুন: "দুঃখিত, এই পণ্যটি আমাদের স্টোরে পাওয়া যায় না।"
৪. যদি তথ্য Knowledge Base-এ না থাকে, বলুন: "দুঃখিত, এই বিষয়ে আমাদের কাছে তথ্য নেই।"
৫. উত্তর সংক্ষিপ্ত, স্পষ্ট ও সহায়ক রাখুন।
৬. পূর্ববর্তী কথোপকথনের সাথে সামঞ্জস্য রেখে উত্তর দিন।
৭. Knowledge Base-এ না থাকলে কোনো ডিসকাউন্ট/বিশেষ মূল্য বা গিফট র্যাপিং সুবিধা উল্লেখ করবেন না।

Knowledge Base:
{kb_context}"""

        # Build messages for LLM
        messages = [{"role": "system", "content": system_prompt}]
        for h in history[-CONTEXT_WINDOW:]:
            messages.append({"role": h["role"], "content": h["content"]})
        messages.append({"role": "user", "content": req.message})

        # Call Groq LLM
        if not groq_client:
            raise HTTPException(500, "GROQ_API_KEY not configured")

        t_gen = time.time()
        try:
            response = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages,
                max_tokens=512,
                temperature=0.3,
            )
            raw_answer = response.choices[0].message.content.strip()
            answer = guard_answer(raw_answer, kb_context)
        except Exception as e:
            raise HTTPException(500, f"LLM error: {str(e)}")
        generation_ms = (time.time() - t_gen) * 1000

    # Save to history
    conn.execute("INSERT INTO chat_history (user_id, role, content) VALUES (?,?,?)",
                 (user_id, "user", req.message))
    conn.execute("INSERT INTO chat_history (user_id, role, content) VALUES (?,?,?)",
                 (user_id, "assistant", answer))
    conn.commit()
    conn.close()

    return {
        "answer": answer,
        "retrieval_ms": round(retrieval_ms, 2),
        "generation_ms": round(generation_ms, 2),
        "kb_hits": len(kb_chunks)
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
    conn.commit()
    conn.close()
    return {"message": "History cleared"}


# ─────────────────────────── Run ───────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
