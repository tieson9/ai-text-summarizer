from fastapi import FastAPI
from fastapi import Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import pipeline
from functools import lru_cache
import logging
from fastapi.concurrency import run_in_threadpool

logging.basicConfig(level=logging.INFO)
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cache the model so it loads only once
@lru_cache()
def get_summarizer():
    return pipeline(
        "summarization",
        model="facebook/bart-large-cnn"   # <-- NEW MODEL HERE
    )

@app.get("/")
def home():
    return {"message": "AI Summarizer API is running successfully"}

@app.head("/")
def home_head():
    return Response(status_code=200)

class SummaryRequest(BaseModel):
    text: str

def summarize_with_bart(text: str) -> str:
    summarizer = get_summarizer()
    result = summarizer(
        text,
        max_length=120,
        min_length=30,
        do_sample=False
    )
    return result[0]["summary_text"]

@app.post("/summarize")
async def summarize_text(req: SummaryRequest):
    logging.info(f"Received text: {req.text}")
    try:
        summary = await run_in_threadpool(summarize_with_bart, req.text)
        return {"summary": summary}
    except Exception as e:
        logging.error("Error during summarization: %s", str(e))
        return {"error": str(e)}

@app.options("/summarize")
def summarize_options():
    return Response(status_code=204)
