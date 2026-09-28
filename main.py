"""
Spam Email Detector - FastAPI app.

  GET  /          -> web UI (paste an email, get a prediction)
  POST /predict   -> JSON API   {"text": "..."}  ->  {"label": "spam", ...}
  GET  /docs      -> auto-generated interactive API docs (Swagger)
  GET  /health    -> health check
"""

from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

BASE_DIR = Path(__file__).parent
pipeline = joblib.load(BASE_DIR / "model.pkl")

app = FastAPI(title="Spam Email Detector", version="1.0.0")


class EmailIn(BaseModel):
    text: str


class PredictionOut(BaseModel):
    label: str
    is_spam: bool
    spam_probability: float
    ham_probability: float


@app.post("/predict", response_model=PredictionOut)
def predict(email: EmailIn):
    text = email.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Email text is empty.")
    proba = pipeline.predict_proba([text])[0]
    is_spam = bool(pipeline.predict([text])[0])
    return PredictionOut(
        label="spam" if is_spam else "not spam",
        is_spam=is_spam,
        spam_probability=round(float(proba[1]), 4),
        ham_probability=round(float(proba[0]), 4),
    )


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def index():
    return HTML_PAGE


HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Spam Email Detector</title>
<style>
  :root { --bg:#f5f7fb; --card:#fff; --text:#1f2937; --muted:#6b7280;
          --ok:#059669; --bad:#dc2626; --accent:#2563eb; --border:#e5e7eb; }
  @media (prefers-color-scheme: dark) {
    :root { --bg:#0f172a; --card:#1e293b; --text:#f1f5f9; --muted:#94a3b8;
            --border:#334155; }
  }
  * { box-sizing:border-box; }
  body { margin:0; font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;
         background:var(--bg); color:var(--text); }
  main { max-width:720px; margin:0 auto; padding:32px 16px; }
  h1 { margin:0 0 4px; font-size:1.6rem; }
  p.sub { margin:0 0 20px; color:var(--muted); }
  .card { background:var(--card); border:1px solid var(--border);
          border-radius:12px; padding:20px; }
  textarea { width:100%; min-height:180px; padding:12px; font:inherit;
             color:var(--text); background:var(--bg);
             border:1px solid var(--border); border-radius:8px; resize:vertical; }
  .row { display:flex; gap:8px; flex-wrap:wrap; margin-top:12px; }
  button { padding:10px 18px; font:inherit; border-radius:8px; cursor:pointer;
           border:1px solid var(--border); background:var(--card); color:var(--text); }
  button.primary { background:var(--accent); border-color:var(--accent); color:#fff; }
  button:disabled { opacity:.6; cursor:wait; }
  #result { margin-top:20px; display:none; }
  .badge { display:inline-block; padding:6px 14px; border-radius:999px;
           font-weight:600; color:#fff; }
  .badge.spam { background:var(--bad); } .badge.ham { background:var(--ok); }
  .bar { height:10px; background:var(--border); border-radius:999px;
         overflow:hidden; margin-top:12px; }
  .bar > div { height:100%; background:var(--bad); width:0; transition:width .4s; }
  .meta { color:var(--muted); font-size:.9rem; margin-top:8px; }
  a { color:var(--accent); }
  footer { margin-top:20px; color:var(--muted); font-size:.9rem; }
</style>
</head>
<body>
<main>
  <h1>📧 Spam Email Detector</h1>
  <p class="sub">Paste an email below and the model will tell you if it looks like spam.</p>
  <div class="card">
    <textarea id="text" placeholder="Paste the email text here..."></textarea>
    <div class="row">
      <button class="primary" id="go">Check email</button>
      <button id="ex1">Spam example</button>
      <button id="ex2">Normal example</button>
    </div>
    <div id="result">
      <span class="badge" id="badge"></span>
      <div class="bar"><div id="fill"></div></div>
      <div class="meta" id="meta"></div>
    </div>
  </div>
  <footer>
    Also available as an API: <a href="/docs">/docs</a> &middot;
    <code>POST /predict</code> with <code>{"text": "..."}</code>
  </footer>
</main>
<script>
const $ = id => document.getElementById(id);
$("ex1").onclick = () => $("text").value =
  "Subject: You won $1,000,000!!! Congratulations! You have been selected to " +
  "receive a cash prize. Click the link now to claim your reward. Limited time offer!";
$("ex2").onclick = () => $("text").value =
  "Subject: Meeting tomorrow\\n\\nHi Ahmed, just a reminder that our project " +
  "meeting is tomorrow at 10 AM. Please bring the updated report. Thanks!";
$("go").onclick = async () => {
  const text = $("text").value.trim();
  if (!text) return;
  $("go").disabled = true; $("go").textContent = "Checking...";
  try {
    const r = await fetch("/predict", {method:"POST",
      headers:{"Content-Type":"application/json"}, body:JSON.stringify({text})});
    if (!r.ok) throw new Error((await r.json()).detail || "Request failed");
    const d = await r.json();
    $("badge").textContent = d.is_spam ? "Spam 🚫" : "Not spam ✅";
    $("badge").className = "badge " + (d.is_spam ? "spam" : "ham");
    $("fill").style.width = (d.spam_probability * 100) + "%";
    $("meta").textContent = "Spam probability: " + (d.spam_probability * 100).toFixed(1) + "%";
    $("result").style.display = "block";
  } catch (e) {
    alert("Error: " + e.message);
  } finally {
    $("go").disabled = false; $("go").textContent = "Check email";
  }
};
</script>
</body>
</html>
"""
