from fastapi import FastAPI
from fastapi import Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import pipeline
from starlette.concurrency import run_in_threadpool
import logging

# -----------------------------------------------------------
# Logging
# -----------------------------------------------------------
logging.basicConfig(level=logging.INFO)

# -----------------------------------------------------------
# App Setup
# -----------------------------------------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # Allow any frontend
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------
# Load summarizer ONCE (critical for Render free tier)
# -----------------------------------------------------------
logging.info("Loading summarization model...")
summarizer = pipeline(
    "summarization",
    model="sshleifer/distilbart-cnn-12-6"   # LIGHT MODEL (works on free tier)
)
logging.info("Model loaded successfully!")

# -----------------------------------------------------------
# Request Model
# -----------------------------------------------------------
class SummaryRequest(BaseModel):
    text: str

# -----------------------------------------------------------
# Summarizer function (thread-safe)
# -----------------------------------------------------------
def summarize_with_bart(text: str) -> str:
    result = summarizer(
        text,
        max_length=120,
        min_length=30,
        do_sample=False
    )
    return result[0]["summary_text"]

# -----------------------------------------------------------
# Routes
# -----------------------------------------------------------
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
        summary = await run_in_threadpool(summarize_with_bart, req.text)
        return {
            "summary": summary,
            "important_sentences": summary.split(". ")[:3]  # optional extra output
        }
    except Exception as e:
        logging.error("Error during summarization: %s", str(e))
        return {"error": str(e)}

@app.options("/summarize")
def summarize_options():
    return Response(status_code=204)