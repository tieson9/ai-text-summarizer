rom fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import pipeline
from starlette.concurrency import run_in_threadpool
import logging

logging.basicConfig(level=logging.INFO)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------
# Load t5-small model (VERY LIGHTWEIGHT - fits Render free)
# -----------------------------------------------------------
logging.info("Loading lightweight summarization model (t5-small)...")

summarizer = pipeline(
    "summarization",
    model="t5-small",
    tokenizer="t5-small"
)

logging.info("Model loaded successfully!")

class SummaryRequest(BaseModel):
    text: str

def summarize_with_t5(text: str) -> str:
    prompt = "summarize: " + text
    result = summarizer(
        prompt,
        max_length=120,
        min_length=30,
        do_sample=False
    )
    return result[0]["summary_text"]

@app.get("/")
def home():
    return {"message": "AI Summarizer API is running successfully"}

@app.head("/")
def home_head():
    return Response(status_code=200)

@app.post("/summarize")
async def summarize_text(req: SummaryRequest):
    logging.info("Received text of length %d", len(req.text))
    try:
        summary = await run_in_threadpool(summarize_with_t5, req.text)
        return {
            "summary": summary,
            "important_sentences": summary.split(". ")[:3]
        }
    except Exception as e:
        logging.error("Error during summarization: %s", str(e))
        return {"error": str(e)}

@app.options("/summarize")
def summarize_options():
    return Response(status_code=204)