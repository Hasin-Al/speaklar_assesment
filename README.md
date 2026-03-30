# শপ সহায়ক — Bengali RAG Chatbot

A production-ready Bengali e-commerce chatbot with RAG (Retrieval-Augmented Generation),
dynamic context management, JWT authentication, and a fast TF-IDF retrieval engine.

---

## Features
- **TF-IDF + Cosine Similarity** retrieval (< 100ms guaranteed)
- **Dynamic context management**: detects self-contained vs follow-up questions
- **Bengali language support** throughout
- **JWT authentication** with SQLite user store
- **Groq LLM** (llama-3.3-70b-versatile) for response generation
- **Chat history** per user with context-window management

---

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Create `.env` file (or export variable)
```bash
export GROQ_API_KEY=your_groq_api_key_here
```
Or create a `.env` file and load it:
```bash
echo "GROQ_API_KEY=your_key" > .env
```

### 3. Place Knowledge Base
Make sure `Knowledge_Bank.txt` is in the same folder as `main.py`.

### 4. Run
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 5. Open browser
- Login: http://localhost:8000/
- Register: http://localhost:8000/register-page
- Chat: http://localhost:8000/chat (after login)

---

## Architecture

```
User Query (Bengali)
        │
        ▼
Context Manager
  ├─ Has own context? → Use query directly
  └─ Follow-up?       → Augment with last user messages
        │
        ▼
TF-IDF Retrieval Engine
  ├─ Tokenize (Bengali + English regex)
  ├─ Cosine similarity against ~5000 KB docs
  └─ Top-5 chunks returned in < 100ms
        │
        ▼
LLM (Groq llama-3.3-70b-versatile)
  ├─ System prompt with KB context
  ├─ Last N turns of conversation history
  └─ Strict "only answer from KB" instruction
        │
        ▼
Bengali Response to User
```

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/register` | Register new user |
| POST | `/api/token` | Login (returns JWT) |
| GET | `/api/me` | Current user info |
| POST | `/api/chat` | Send message, get RAG response |
| GET | `/api/history` | Get chat history |
| DELETE | `/api/history` | Clear chat history |

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GROQ_API_KEY` | Yes | Your Groq API key |

---

## How Context Management Works

1. **Self-contained query**: "স্মার্টফোনের দাম কত?" — goes directly to RAG
2. **Follow-up query**: "এর ওয়ারেন্টি আছে?" — the system prepends the last user message
   to the RAG query: "স্মার্টফোনের দাম কত? এর ওয়ারেন্টি আছে?" ensuring proper retrieval
3. **Conversation history**: Last 5 turns are always passed to the LLM for coherent responses

---

## Performance

- TF-IDF index is built once at startup (~2–5 seconds for ~5000 docs)
- Each retrieval query runs in **< 50ms** (pure NumPy matrix multiplication)
- No external embedding API calls for retrieval
