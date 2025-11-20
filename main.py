from fastapi import FastAPI
from transformers import pipeline

app = FastAPI()
summarizer = pipeline("summarization")

@app.get("/")
def home():
    return {"message": "AI Summarizer API is running"}

@app.post("/summarize")
def summarize_text(text: str):
    summary = summarizer(text, max_length=100, min_length=30, do_sample=False)
    return {"summary": summary[0]["summary_text"]}
