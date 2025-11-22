from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from transformers import pipeline
from functools import lru_cache

app = FastAPI()

@lru_cache()
def get_summarizer():
    return pipeline("summarization", model="sshleifer/distilbart-cnn-6-6")

@app.get("/", response_class=HTMLResponse)
def home():
    return (
        """
        <!DOCTYPE html>
        <html lang="en">
        <head>
          <meta charset="UTF-8" />
          <meta name="viewport" content="width=device-width, initial-scale=1.0" />
          <title>AI Text Summarizer API</title>
          <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap" rel="stylesheet">
          <style>
            :root { --accent: #2563EB; }
            * { box-sizing: border-box; }
            body { margin: 0; font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; color: #111827; background: #ffffff; }
            .container { max-width: 960px; margin: 0 auto; padding: 0 24px; }
            header { width: 100%; border-bottom: 1px solid #e5e7eb; }
            header .container { padding: 40px 24px; text-align: center; }
            h1 { margin: 0; font-size: 2rem; font-weight: 600; }
            .subtitle { margin-top: 8px; color: #4b5563; }
            .hero { text-align: center; padding: 40px 24px; }
            .hero p { font-size: 1.125rem; color: #374151; }
            .btn { display: inline-block; margin-top: 16px; background: var(--accent); color: #fff; padding: 12px 20px; border-radius: 8px; text-decoration: none; font-weight: 600; }
            .section { padding: 40px 0; }
            .features { display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; }
            .card { border: 1px solid #e5e7eb; border-radius: 8px; padding: 16px; text-align: center; }
            .code { background: #0f172a; color: #e5e7eb; border-radius: 8px; padding: 16px; overflow-x: auto; }
            .code pre { margin: 0; white-space: pre-wrap; }
            .form { border: 1px solid #e5e7eb; border-radius: 8px; padding: 16px; background: #f9fafb; }
            textarea { width: 100%; min-height: 160px; padding: 12px; border-radius: 8px; border: 1px solid #d1d5db; font-family: inherit; font-size: 1rem; }
            .form .actions { margin-top: 12px; display: flex; gap: 12px; align-items: center; }
            .result { margin-top: 12px; background: #fff; border: 1px solid #e5e7eb; border-radius: 8px; padding: 12px; min-height: 48px; }
            footer { text-align: center; color: #6b7280; padding: 24px; border-top: 1px solid #e5e7eb; }
            @media (max-width: 768px) { .features { grid-template-columns: 1fr; } }
          </style>
        </head>
        <body>
          <header>
            <div class="container">
              <h1>AI Text Summarizer API</h1>
              <div class="subtitle">Fast, accurate, production-ready text summarization.</div>
            </div>
          </header>

          <main class="container">
            <section class="hero">
              <p>Turn long text into clean summaries with one API request.</p>
              <a href="#test" class="btn">▶ Try the API</a>
            </section>

            <section class="section">
              <div class="features">
                <div class="card">⚡ Fast summarization using transformer models</div>
                <div class="card">🔍 Accurate results even for long text</div>
                <div class="card">🚀 Free to use for demos and testing</div>
              </div>
            </section>

            <section class="section">
              <h2>API Documentation Preview</h2>
              <div class="code">
                <pre><code>POST https://your-render-url/summarize
Body:
{
  "text": "Your long text here"
}
Response:
{
  "summary": "Short summary..."
}</code></pre>
              </div>
            </section>

            <section id="test" class="section">
              <h2>Test the API</h2>
              <div class="form">
                <label for="input">Text</label>
                <textarea id="input" placeholder="Paste long text here..."></textarea>
                <div class="actions">
                  <button id="run" class="btn">Summarize</button>
                  <span id="status"></span>
                </div>
                <div class="result" id="result"></div>
              </div>
            </section>
          </main>

          <footer>
            © 2024 tieson98 – AI Text Summarizer API · Built with FastAPI and Render
          </footer>

          <script>
            const runBtn = document.getElementById('run');
            const input = document.getElementById('input');
            const status = document.getElementById('status');
            const result = document.getElementById('result');

            runBtn.addEventListener('click', async () => {
              const text = input.value.trim();
              if (!text) { result.textContent = 'Enter text to summarize.'; return; }
              status.textContent = 'Summarizing…';
              result.textContent = '';
              try {
                const res = await fetch('/summarize', {
                  method: 'POST',
                  headers: { 'Content-Type': 'application/json' },
                  body: JSON.stringify({ text })
                });
                if (!res.ok) { throw new Error('Request failed'); }
                const data = await res.json();
                result.textContent = data.summary ?? JSON.stringify(data);
              } catch (e) {
                result.textContent = 'Error running summarization.';
              } finally {
                status.textContent = '';
              }
            });
          </script>
        </body>
        </html>
        """
    )

class SummarizeRequest(BaseModel):
    text: str

@app.post("/summarize")
def summarize_text(payload: SummarizeRequest):
    summarizer = get_summarizer()
    summary = summarizer(payload.text, max_length=100, min_length=30, do_sample=False)
    return {"summary": summary[0]["summary_text"]}
