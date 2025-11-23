from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from transformers import pipeline
from functools import lru_cache

app = FastAPI()

@lru_cache()
def get_summarizer():
    return pipeline("summarization", model="sshleifer/distilbart-cnn-6-6")

@app.get("/")
def home():
    return {"message": "AI Summarizer API is running"}

class SummarizeRequest(BaseModel):
    text: str

@app.post("/summarize")
def summarize_text(payload: SummarizeRequest):
    summarizer = get_summarizer()
    summary = summarizer(payload.text, max_length=100, min_length=30, do_sample=False)
    return {"summary": summary[0]["summary_text"]}
