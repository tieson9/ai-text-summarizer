from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import pipeline
from functools import lru_cache

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
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

class SummarizeRequest(BaseModel):
    text: str

@app.post("/summarize")
def summarize_text(payload: SummarizeRequest):
    summarizer = get_summarizer()
    summary = summarizer(
        payload.text,
        max_length=120,
        min_length=40,
        do_sample=False
    )
    return {"summary": summary[0]["summary_text"]}