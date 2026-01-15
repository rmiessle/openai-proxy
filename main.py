import os
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

# --- Config (set these as env vars on Reclaim) ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set")

ALLOWED_MODELS = set(os.getenv("ALLOWED_MODELS", "gpt-5.2-pro").split(","))
WORKSHOP_CODE = os.getenv("WORKSHOP_CODE")  # optional shared code
MAX_CHARS = int(os.getenv("MAX_CHARS", "12000"))

# Binder origins vary; start permissive and rely on rate limits + optional code.
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")

client = OpenAI(api_key=OPENAI_API_KEY)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

class ChatIn(BaseModel):
    messages: list  # [{"role":"user","content":"..."}, ...]
    model: str = "gpt-4.1-mini"
    workshop_code: str | None = None

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/chat")
def chat(payload: ChatIn, request: Request):
    if payload.model not in ALLOWED_MODELS:
        raise HTTPException(400, "Model not allowed")

    if WORKSHOP_CODE and payload.workshop_code != WORKSHOP_CODE:
        raise HTTPException(401, "Invalid workshop code")

    total = sum(len(m.get("content", "")) for m in payload.messages if isinstance(m, dict))
    if total > MAX_CHARS:
        raise HTTPException(413, "Message too large")

    resp = client.responses.create(
        model=payload.model,
        input=payload.messages,
    )
    return {"text": resp.output_text}
