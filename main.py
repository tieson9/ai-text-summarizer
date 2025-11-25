from typing import List, Literal
import os
import re
import logging
from fastapi import FastAPI, Response, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from starlette.concurrency import run_in_threadpool

logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="Universal AI Summarizer API",
    description="Provider-agnostic summarization backend for OpenAI and Google Generative AI.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SummarizePayload(BaseModel):
    """Request payload schema for summarization."""

    text: str = Field(..., min_length=5, description="Input text to summarize")
    provider: Literal["openai", "google"] = Field(..., description="AI provider")
    model: str = Field(..., min_length=1, description="Model identifier for the provider")
    api_key: str = Field(..., min_length=10, description="API key for the provider")

    @validator("text")
    def validate_text(cls, v: str) -> str:
        t = v.strip()
        if len(t) < 5:
            raise ValueError("text must be at least 5 characters")
        if len(t) > 100_000:
            raise ValueError("text too long; limit to 100k characters")
        return t

class SummarizeResponse(BaseModel):
    """Response schema for summarization."""

    summary: str
    important_sentences: List[str]

def _extract_sentences(text: str, limit: int = 3) -> List[str]:
    s = re.split(r"(?<=[\.!?])\s+", text.strip())
    return [x.strip() for x in s if x.strip()][:limit]

def _summarize_with_openai(text: str, model: str, api_key: str) -> str:
    try:
        import openai  # type: ignore
    except Exception:
        raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail="openai library is not installed")

    try:
        if hasattr(openai, "ChatCompletion"):
            openai.api_key = api_key
            prompt = (
                "You are a concise summarizer. Summarize the following text in a clear, short paragraph.\n\n"
                f"Text:\n{text}"
            )
            resp = openai.ChatCompletion.create(
                model=model,
                messages=[{"role": "system", "content": "Summarize input clearly and concisely."},
                          {"role": "user", "content": prompt}],
                max_tokens=256,
                temperature=0.2,
            )
            return resp["choices"][0]["message"]["content"].strip()
        else:
            from openai import OpenAI  # type: ignore

            client = OpenAI(api_key=api_key)
            prompt = (
                "You are a concise summarizer. Summarize the following text in a clear, short paragraph.\n\n"
                f"Text:\n{text}"
            )
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": "Summarize input clearly and concisely."},
                          {"role": "user", "content": prompt}],
                max_tokens=256,
                temperature=0.2,
            )
            return resp.choices[0].message.content.strip()
    except Exception as e:
        msg = str(e)
        if "Incorrect API key" in msg or "authentication" in msg.lower():
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="OpenAI authentication failed")
        if "rate limit" in msg.lower():
            raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="OpenAI rate limit exceeded")
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=f"OpenAI error: {msg}")

def _summarize_with_google(text: str, model: str, api_key: str) -> str:
    try:
        import google.generativeai as genai  # type: ignore
    except Exception:
        raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail="google-generativeai library is not installed")

    try:
        genai.configure(api_key=api_key)
        gm = genai.GenerativeModel(model)
        prompt = (
            "You are a concise summarizer. Summarize the following text in a clear, short paragraph.\n\n"
            f"Text:\n{text}"
        )
        resp = gm.generate_content(prompt)
        content = getattr(resp, "text", None) or (resp.candidates[0].content.parts[0].text if resp.candidates else "")
        if not content:
            raise RuntimeError("Empty response from Google Generative AI")
        return content.strip()
    except Exception as e:
        msg = str(e)
        if "API key" in msg or "unauthorized" in msg.lower():
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Google authentication failed")
        if "quota" in msg.lower() or "rate" in msg.lower():
            raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="Google rate limit exceeded")
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=f"Google error: {msg}")

@app.get("/", response_model=dict, tags=["health"])
def health() -> dict:
    """Health check endpoint returning a simple status."""
    return {"status": "ok"}

@app.head("/", tags=["health"])
def health_head() -> Response:
    """Lightweight health check using HEAD."""
    return Response(status_code=status.HTTP_200_OK)

@app.options("/summarize", tags=["summarize"])
def summarize_options() -> Response:
    """CORS preflight endpoint for summarize route."""
    return Response(status_code=status.HTTP_204_NO_CONTENT)

@app.post("/summarize", response_model=SummarizeResponse, tags=["summarize"])
async def summarize(payload: SummarizePayload) -> SummarizeResponse:
    """Summarize input text using the requested provider and model."""
    if not payload.api_key or len(payload.api_key.strip()) < 10:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid API key format")

    if payload.provider == "openai":
        summary_text = await run_in_threadpool(_summarize_with_openai, payload.text, payload.model, payload.api_key)
    elif payload.provider == "google":
        summary_text = await run_in_threadpool(_summarize_with_google, payload.text, payload.model, payload.api_key)
    else:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Unsupported provider")

    sentences = _extract_sentences(summary_text, limit=3)
    return SummarizeResponse(summary=summary_text, important_sentences=sentences)

def _unit_test_scaffold() -> None:
    """Minimal inline test scaffold for quick verification.

    Run with pytest by importing this module and invoking functions.
    """
    sample = "This is a test. It has two sentences. Summaries should be concise."
    assert len(_extract_sentences(sample, 3)) == 3
