from typing import Literal
import logging
import httpx
from fastapi import FastAPI, Response, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator

logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="Universal AI Summarizer API",
    description="Provider-agnostic summarization backend for OpenAI, Google Gemini, Groq, DeepSeek, and Mistral.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ProviderPayload(BaseModel):
    """Request payload for provider tests and summarization."""

    provider: Literal["openai", "google", "groq", "deepseek", "mistral"] = Field(..., description="AI provider")
    model: str = Field(..., min_length=1, description="Model identifier for the provider")
    api_key: str = Field(..., min_length=10, description="API key for the provider")
    text: str = Field(..., min_length=1, description="Input text, use 'Ping' for test")

    @validator("text")
    def validate_text(cls, v: str) -> str:
        t = v.strip()
        if not t:
            raise ValueError("text must not be empty")
        if len(t) > 100_000:
            raise ValueError("text too long; limit to 100k characters")
        return t

class ProviderTestResponse(BaseModel):
    """Response model for provider health test."""

    provider_status: str | None = None
    error: str | None = None

class SummarizeResponse(BaseModel):
    """Response model for summarization."""

    summary: str

async def _call_openai_chat(model: str, api_key: str, messages: list[dict]) -> str:
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    body = {"model": model, "messages": messages, "max_tokens": 256, "temperature": 0.2}
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.post(url, headers=headers, json=body)
    if r.status_code == 401:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="OpenAI authentication failed")
    if r.status_code == 429:
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="OpenAI rate limit exceeded")
    if r.is_error:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=f"OpenAI error: {r.text}")
    data = r.json()
    return data["choices"][0]["message"]["content"].strip()

async def _call_groq_chat(model: str, api_key: str, messages: list[dict]) -> str:
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    body = {"model": model, "messages": messages, "max_tokens": 256, "temperature": 0.2}
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.post(url, headers=headers, json=body)
    if r.status_code == 401:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Groq authentication failed")
    if r.status_code == 429:
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="Groq rate limit exceeded")
    if r.is_error:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=f"Groq error: {r.text}")
    data = r.json()
    return data["choices"][0]["message"]["content"].strip()

async def _call_deepseek_chat(model: str, api_key: str, messages: list[dict]) -> str:
    url = "https://api.deepseek.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    body = {"model": model, "messages": messages, "max_tokens": 256, "temperature": 0.2}
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.post(url, headers=headers, json=body)
    if r.status_code == 401:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="DeepSeek authentication failed")
    if r.status_code == 429:
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="DeepSeek rate limit exceeded")
    if r.is_error:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=f"DeepSeek error: {r.text}")
    data = r.json()
    return data["choices"][0]["message"]["content"].strip()

async def _call_mistral_chat(model: str, api_key: str, messages: list[dict]) -> str:
    url = "https://api.mistral.ai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    body = {"model": model, "messages": messages, "max_tokens": 256, "temperature": 0.2}
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.post(url, headers=headers, json=body)
    if r.status_code == 401:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Mistral authentication failed")
    if r.status_code == 429:
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="Mistral rate limit exceeded")
    if r.is_error:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=f"Mistral error: {r.text}")
    data = r.json()
    return data["choices"][0]["message"]["content"].strip()

async def _call_google_generate(model: str, api_key: str, text: str) -> str:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    params = {"key": api_key}
    body = {"contents": [{"parts": [{"text": text}]}]}
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.post(url, params=params, json=body)
    if r.status_code == 401:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Google authentication failed")
    if r.status_code == 429:
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="Google rate limit exceeded")
    if r.is_error:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=f"Google error: {r.text}")
    data = r.json()
    if "candidates" in data and data["candidates"]:
        candidate = data["candidates"][0]
        parts = candidate.get("content", {}).get("parts") or candidate.get("content", {}).get("parts", [])
        if parts:
            for p in parts:
                if "text" in p:
                    return p["text"].strip()
    if "text" in data:
        return str(data["text"]).strip()
    raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="Google response missing text")

@app.get("/", response_model=dict, tags=["health"])
def health() -> dict:
    """Health check endpoint."""
    return {"status": "ok"}

@app.head("/", tags=["health"])
def health_head() -> Response:
    """HEAD health check."""
    return Response(status_code=status.HTTP_200_OK)

@app.options("/summarize", tags=["summarize"])
def summarize_options() -> Response:
    """CORS preflight endpoint."""
    return Response(status_code=status.HTTP_204_NO_CONTENT)

@app.post("/test-provider", response_model=ProviderTestResponse, tags=["provider"])
async def test_provider(payload: ProviderPayload) -> ProviderTestResponse:
    """Makes a minimal provider request to verify credentials and connectivity."""
    try:
        if payload.provider == "openai":
            _ = await _call_openai_chat(payload.model, payload.api_key, [{"role": "user", "content": payload.text}])
        elif payload.provider == "groq":
            _ = await _call_groq_chat(payload.model, payload.api_key, [{"role": "user", "content": payload.text}])
        elif payload.provider == "deepseek":
            _ = await _call_deepseek_chat(payload.model, payload.api_key, [{"role": "user", "content": payload.text}])
        elif payload.provider == "mistral":
            _ = await _call_mistral_chat(payload.model, payload.api_key, [{"role": "user", "content": payload.text}])
        elif payload.provider == "google":
            _ = await _call_google_generate(payload.model, payload.api_key, payload.text)
        else:
            return ProviderTestResponse(error="Unsupported provider")
        return ProviderTestResponse(provider_status="OK")
    except HTTPException as he:
        return ProviderTestResponse(error=he.detail)
    except Exception:
        return ProviderTestResponse(error="Provider request failed")

@app.post("/summarize", response_model=SummarizeResponse, tags=["summarize"])
async def summarize(payload: ProviderPayload) -> SummarizeResponse:
    """Summarizes input text using the selected provider and model."""
    if not payload.api_key or len(payload.api_key.strip()) < 10:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid API key format")
    try:
        if payload.provider == "openai":
            summary_text = await _call_openai_chat(
                payload.model,
                payload.api_key,
                [{"role": "system", "content": "Summarize clearly and concisely."}, {"role": "user", "content": payload.text}],
            )
        elif payload.provider == "groq":
            summary_text = await _call_groq_chat(
                payload.model,
                payload.api_key,
                [{"role": "system", "content": "Summarize clearly and concisely."}, {"role": "user", "content": payload.text}],
            )
        elif payload.provider == "deepseek":
            summary_text = await _call_deepseek_chat(
                payload.model,
                payload.api_key,
                [{"role": "system", "content": "Summarize clearly and concisely."}, {"role": "user", "content": payload.text}],
            )
        elif payload.provider == "mistral":
            summary_text = await _call_mistral_chat(
                payload.model,
                payload.api_key,
                [{"role": "system", "content": "Summarize clearly and concisely."}, {"role": "user", "content": payload.text}],
            )
        elif payload.provider == "google":
            summary_text = await _call_google_generate(payload.model, payload.api_key, payload.text)
        else:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Unsupported provider")
    except HTTPException as he:
        raise he
    except Exception:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="Provider request failed")
    return SummarizeResponse(summary=summary_text)
