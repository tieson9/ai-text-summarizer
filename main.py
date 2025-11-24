from fastapi import FastAPI
from fastapi import Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import pipeline
from functools import lru_cache
import logging

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

class SummarizeRequest(BaseModel):
    text: str

@app.post("/summarize")
def summarize_text(payload: SummarizeRequest):
    logging.basicConfig(level=logging.INFO)
    logging.info("Received text: %s", payload.text[:200])
    try:
        summarizer = get_summarizer()
        logging.info("Model loaded successfully")
        result = summarizer(
            payload.text,
            max_length=120,
            min_length=30,
            do_sample=False
        )
        logging.info("Summarization completed")
        logging.info("Summary: %s", result[0]["summary_text"])
        return {
            "summary": result[0]["summary_text"],
            "important_sentences": result[0]["summary_text"].split(". ")[:3]
        }
    except Exception as e:
        logging.error("Error during summarization: %s", str(e))
        return {"error": str(e)}

@app.options("/summarize")
def summarize_options():
    return Response(status_code=204)
