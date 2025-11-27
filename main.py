from typing import Literal, Optional, Dict, Any
import logging
import httpx
from fastapi import FastAPI, Response, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator, AnyHttpUrl
import time
from xml.etree import ElementTree as ET

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
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.post(url, headers=headers, json=body)
    except httpx.TimeoutException:
        raise HTTPException(status.HTTP_408_REQUEST_TIMEOUT, "OpenAI timeout")
    except httpx.RequestError as e:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, f"OpenAI network error: {e}")

    if r.status_code == 401:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid OpenAI API key")
    if r.status_code == 404:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Model '{model}' not found on OpenAI")
    if r.status_code == 429:
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, "OpenAI rate limit exceeded")
    if r.is_error:
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, f"OpenAI error: {r.text}")

    try:
        data = r.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception:
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, f"OpenAI returned invalid JSON: {r.text}")

async def _call_groq_chat(model: str, api_key: str, messages: list[dict]) -> str:
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    body = {"model": model, "messages": messages, "max_tokens": 256, "temperature": 0.2}
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.post(url, headers=headers, json=body)
    except httpx.TimeoutException:
        raise HTTPException(status.HTTP_408_REQUEST_TIMEOUT, "Groq timeout")
    except httpx.RequestError as e:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, f"Groq network error: {e}")

    if r.status_code == 401:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid Groq API key")
    if r.status_code == 404:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Model '{model}' not found on Groq")
    if r.status_code == 429:
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, "Groq rate limit exceeded")
    if r.is_error:
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, f"Groq error: {r.text}")

    try:
        data = r.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception:
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, f"Groq returned invalid JSON: {r.text}")

async def _call_deepseek_chat(model: str, api_key: str, messages: list[dict]) -> str:
    url = "https://api.deepseek.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    body = {"model": model, "messages": messages, "max_tokens": 256, "temperature": 0.2}
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.post(url, headers=headers, json=body)
    except httpx.TimeoutException:
        raise HTTPException(status.HTTP_408_REQUEST_TIMEOUT, "DeepSeek timeout")
    except httpx.RequestError as e:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, f"DeepSeek network error: {e}")

    if r.status_code == 401:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid DeepSeek API key")
    if r.status_code == 404:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Model '{model}' not found on DeepSeek")
    if r.status_code == 429:
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, "DeepSeek rate limit exceeded")
    if r.is_error:
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, f"DeepSeek error: {r.text}")

    try:
        data = r.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception:
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, f"DeepSeek returned invalid JSON: {r.text}")

async def _call_mistral_chat(model: str, api_key: str, messages: list[dict]) -> str:
    url = "https://api.mistral.ai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    body = {"model": model, "messages": messages, "max_tokens": 256, "temperature": 0.2}
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.post(url, headers=headers, json=body)
    except httpx.TimeoutException:
        raise HTTPException(status.HTTP_408_REQUEST_TIMEOUT, "Mistral timeout")
    except httpx.RequestError as e:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, f"Mistral network error: {e}")

    if r.status_code == 401:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid Mistral API key")
    if r.status_code == 404:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Model '{model}' not found on Mistral")
    if r.status_code == 429:
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, "Mistral rate limit exceeded")
    if r.is_error:
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, f"Mistral error: {r.text}")

    try:
        data = r.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception:
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, f"Mistral returned invalid JSON: {r.text}")

async def _call_google_generate(model: str, api_key: str, text: str) -> str:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    params = {"key": api_key}
    body = {"contents": [{"parts": [{"text": text}]}]}
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.post(url, params=params, json=body)
    except httpx.TimeoutException:
        raise HTTPException(status.HTTP_408_REQUEST_TIMEOUT, "Google Gemini timeout")
    except httpx.RequestError as e:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, f"Google network error: {e}")

    if r.status_code == 401:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid Google API key")
    if r.status_code == 404:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Model '{model}' not found on Google Gemini")
    if r.status_code == 429:
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, "Google rate limit exceeded")
    if r.is_error:
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, f"Google error: {r.text}")

    try:
        data = r.json()
    except Exception:
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, f"Google returned invalid JSON: {r.text}")

    # Extract text safely across candidate formats
    try:
        if isinstance(data, dict):
            candidates = data.get("candidates")
            if candidates:
                cand = candidates[0]
                content = cand.get("content") or {}
                parts = content.get("parts") or []
                for p in parts:
                    t = p.get("text")
                    if t:
                        return str(t).strip()
        t2 = data.get("text")
        if t2:
            return str(t2).strip()
    except Exception:
        pass
    raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Google response missing text")

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

 

@app.post("/summarize", response_model=SummarizeResponse, tags=["summarize"])
async def summarize(payload: ProviderPayload) -> SummarizeResponse:
    """Summarizes input text using the selected provider and model with clean error handling."""

    if not payload.api_key or len(payload.api_key.strip()) < 10:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid API key format",
        )

    try:
        if payload.provider == "openai":
            summary_text = await _call_openai_chat(
                payload.model,
                payload.api_key,
                [
                    {"role": "system", "content": "Summarize clearly and concisely."},
                    {"role": "user", "content": payload.text},
                ],
            )
        elif payload.provider == "groq":
            summary_text = await _call_groq_chat(
                payload.model,
                payload.api_key,
                [
                    {"role": "system", "content": "Summarize clearly and concisely."},
                    {"role": "user", "content": payload.text},
                ],
            )
        elif payload.provider == "deepseek":
            summary_text = await _call_deepseek_chat(
                payload.model,
                payload.api_key,
                [
                    {"role": "system", "content": "Summarize clearly and concisely."},
                    {"role": "user", "content": payload.text},
                ],
            )
        elif payload.provider == "mistral":
            summary_text = await _call_mistral_chat(
                payload.model,
                payload.api_key,
                [
                    {"role": "system", "content": "Summarize clearly and concisely."},
                    {"role": "user", "content": payload.text},
                ],
            )
        elif payload.provider == "google":
            summary_text = await _call_google_generate(
                payload.model, payload.api_key, payload.text
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Unsupported provider",
            )

    except HTTPException as he:
        raise he
    except httpx.TimeoutException:
        raise HTTPException(
            status_code=status.HTTP_408_REQUEST_TIMEOUT,
            detail="Provider timeout – the AI model took too long to respond.",
        )
    except httpx.RequestError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Network error contacting provider: {str(e)}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected provider error: {str(e)}",
        )

    return SummarizeResponse(summary=summary_text)

@app.post("/test-provider")
async def test_provider(req: dict):
    provider = req.get("provider")
    api_key = req.get("api_key")
    model = req.get("model")
    if not provider or not api_key or not model:
        return {"error": "provider, model, and api_key are required"}
    try:
        p = str(provider).lower()
        if p == "openai":
            return await test_openai(api_key, model)
        elif p in ("gemini", "google"):
            return await test_gemini(api_key, model)
        elif p == "groq":
            return await test_groq(api_key, model)
        elif p == "deepseek":
            return await test_deepseek(api_key, model)
        else:
            return {"error": "Unknown provider"}
    except Exception as e:
        return {"error": str(e)}

async def test_openai(api_key: str, model: str) -> dict:
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": [{"role": "user", "content": "ping"}], "max_tokens": 10}
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.post(url, json=payload, headers=headers)
    if r.status_code == 200:
        return {"provider": "openai", "status": "OK"}
    return {"provider": "openai", "error": r.text} 

async def test_gemini(api_key: str, model: str) -> dict:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    params = {"key": api_key}
    payload = {"contents": [{"parts": [{"text": "ping"}]}]}
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.post(url, params=params, json=payload)
    if r.status_code == 200:
        return {"provider": "gemini", "status": "OK"}
    return {"provider": "gemini", "error": r.text}

async def test_groq(api_key: str, model: str) -> dict:
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": [{"role": "user", "content": "ping"}], "max_tokens": 10}
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.post(url, json=payload, headers=headers)
    if r.status_code == 200:
        return {"provider": "groq", "status": "OK"}
    return {"provider": "groq", "error": r.text}

async def test_deepseek(api_key: str, model: str) -> dict:
    url = "https://api.deepseek.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": [{"role": "user", "content": "ping"}], "max_tokens": 10}
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.post(url, json=payload, headers=headers)
    if r.status_code == 200:
        return {"provider": "deepseek", "status": "OK"}
    return {"provider": "deepseek", "error": r.text}

class APITestRequest(BaseModel):
    url: AnyHttpUrl
    method: Literal["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"]
    headers: Optional[Dict[str, str]] = None
    params: Optional[Dict[str, Any]] = None
    body: Optional[Any] = None
    auth_type: Literal["none", "bearer", "basic"] = "none"
    bearer_token: Optional[str] = None
    basic_username: Optional[str] = None
    basic_password: Optional[str] = None
    graphql: bool = False
    graphql_query: Optional[str] = None
    graphql_variables: Optional[Dict[str, Any]] = None
    timeout_sec: float = 15.0

    @validator("timeout_sec")
    def _valid_timeout(cls, v: float) -> float:
        if v <= 0 or v > 120:
            raise ValueError("timeout_sec must be between 0 and 120")
        return v

@app.post("/test-api", tags=["testing"])
async def test_api(req: APITestRequest) -> Dict[str, Any]:
    hdrs: Dict[str, str] = dict(req.headers or {})
    auth = None
    if req.auth_type == "bearer":
        if not req.bearer_token:
            return {"success": False, "error": "missing bearer_token"}
        hdrs["Authorization"] = f"Bearer {req.bearer_token}"
    elif req.auth_type == "basic":
        if not req.basic_username or not req.basic_password:
            return {"success": False, "error": "missing basic credentials"}
        auth = (req.basic_username, req.basic_password)

    json_body = None
    data_body = None
    if req.graphql:
        if not req.graphql_query:
            return {"success": False, "error": "missing graphql_query"}
        hdrs.setdefault("Content-Type", "application/json")
        json_body = {"query": req.graphql_query, "variables": req.graphql_variables or {}}
        method = "POST"
    else:
        method = req.method
        if isinstance(req.body, dict):
            hdrs.setdefault("Content-Type", "application/json")
            json_body = req.body
        elif isinstance(req.body, str):
            data_body = req.body

    start = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=req.timeout_sec) as client:
            r = await client.request(
                method,
                str(req.url),
                headers=hdrs,
                params=req.params,
                json=json_body,
                data=data_body,
                auth=auth,
            )
    except httpx.TimeoutException:
        return {"success": False, "error": "timeout"}
    except httpx.RequestError as e:
        return {"success": False, "error": f"request_error: {str(e)}"}
    elapsed_ms = round((time.perf_counter() - start) * 1000, 2)

    ctype = r.headers.get("Content-Type", "")
    lower_ctype = ctype.lower()
    json_valid = False
    xml_valid = False
    graphql_valid = False
    parsed_preview = None

    if "json" in lower_ctype:
        try:
            obj = r.json()
            json_valid = True
            if req.graphql:
                graphql_valid = isinstance(obj, dict) and ("data" in obj) and ("errors" not in obj)
            parsed_preview = obj if isinstance(obj, dict) else obj
        except Exception:
            json_valid = False
    elif "xml" in lower_ctype:
        try:
            root = ET.fromstring(r.text)
            xml_valid = True
            parsed_preview = root.tag
        except Exception:
            xml_valid = False
    else:
        parsed_preview = (r.text or "")[:500]

    ok_status = 200 <= r.status_code < 300
    success = ok_status and ((json_valid and (not req.graphql or graphql_valid)) or xml_valid or (parsed_preview is not None))

    return {
        "success": success,
        "status_code": r.status_code,
        "response_time_ms": elapsed_ms,
        "content_type": ctype,
        "headers": dict(r.headers),
        "validation": {
            "json_valid": json_valid,
            "xml_valid": xml_valid,
            "graphql_valid": graphql_valid,
        },
        "body_preview": parsed_preview if isinstance(parsed_preview, str) else str(parsed_preview)[:500],
    }
