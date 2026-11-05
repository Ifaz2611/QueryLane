"""Lightweight zero-dependency web demo for QueryLane.

Runs an HTTP server on localhost that exposes QueryLane's routing and lane
explainer through a clean, interactive browser interface with visual lane diagrams
and sample presets.
"""

from __future__ import annotations

import json
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

from agent.explain import explain_decision

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>QueryLane -- Interactive Route & Lane Explainer</title>
<style>
  :root {
    --bg: #0f172a;
    --card: #1e293b;
    --border: #334155;
    --text: #f1f5f9;
    --muted: #94a3b8;
    --primary: #38bdf8;
    --primary-hover: #0ea5e9;
    --accent: #10b981;
    --warning: #f59e0b;
    --danger: #ef4444;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: var(--bg);
    color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    line-height: 1.5;
    padding: 2rem 1rem;
    min-height: 100vh;
  }
  .container {
    max-width: 900px;
    margin: 0 auto;
  }
  header {
    margin-bottom: 2rem;
    text-align: center;
  }
  header h1 {
    font-size: 2.25rem;
    font-weight: 800;
    letter-spacing: -0.025em;
    color: #fff;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.75rem;
  }
  header p {
    color: var(--muted);
    font-size: 1.05rem;
    margin-top: 0.5rem;
  }
  .card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
  }
  .form-group {
    margin-bottom: 1rem;
  }
  label {
    display: block;
    font-size: 0.875rem;
    font-weight: 600;
    margin-bottom: 0.5rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }
  textarea {
    width: 100%;
    min-height: 110px;
    padding: 0.85rem;
    background: #090d16;
    border: 1px solid var(--border);
    border-radius: 8px;
    color: #fff;
    font-size: 1rem;
    font-family: inherit;
    resize: vertical;
  }
  textarea:focus {
    outline: none;
    border-color: var(--primary);
  }
  .presets {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin-bottom: 1.25rem;
  }
  .preset-btn {
    background: #273549;
    color: var(--text);
    border: 1px solid var(--border);
    padding: 0.35rem 0.75rem;
    border-radius: 20px;
    font-size: 0.8rem;
    cursor: pointer;
    transition: all 0.15s ease;
  }
  .preset-btn:hover {
    background: var(--primary);
    color: #000;
    border-color: var(--primary);
  }
  .btn-submit {
    background: var(--primary);
    color: #04101d;
    font-weight: 700;
    font-size: 1rem;
    padding: 0.75rem 1.75rem;
    border: none;
    border-radius: 8px;
    cursor: pointer;
    transition: background 0.15s ease;
    width: 100%;
  }
  .btn-submit:hover {
    background: var(--primary-hover);
  }
  .lane-pipeline {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 0.75rem;
    margin: 1.5rem 0;
  }
  @media (max-width: 680px) {
    .lane-pipeline { grid-template-columns: 1fr; }
  }
  .lane-box {
    background: #131d2e;
    border: 2px solid var(--border);
    border-radius: 8px;
    padding: 1rem 0.75rem;
    text-align: center;
    transition: all 0.2s ease;
  }
  .lane-box.active {
    border-color: var(--accent);
    background: rgba(16, 185, 129, 0.12);
    box-shadow: 0 0 15px rgba(16, 185, 129, 0.3);
  }
  .lane-box .lane-title {
    font-weight: 700;
    font-size: 0.95rem;
    margin-bottom: 0.25rem;
  }
  .lane-box .lane-cost {
    font-size: 0.75rem;
    color: var(--muted);
  }
  .badge {
    display: inline-block;
    padding: 0.25rem 0.65rem;
    border-radius: 6px;
    font-weight: 700;
    font-size: 0.85rem;
    text-transform: uppercase;
  }
  .badge-specialist { background: rgba(16, 185, 129, 0.2); color: #34d399; }
  .badge-local { background: rgba(56, 189, 248, 0.2); color: #38bdf8; }
  .badge-cloud { background: rgba(245, 158, 11, 0.2); color: #fbbf24; }
  .stat-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1rem;
    margin-bottom: 1.5rem;
  }
  @media (max-width: 600px) {
    .stat-grid { grid-template-columns: 1fr; }
  }
  .stat-card {
    background: #131d2e;
    padding: 1rem;
    border-radius: 8px;
    border: 1px solid var(--border);
  }
  .stat-val {
    font-size: 1.35rem;
    font-weight: 800;
    color: #fff;
    margin-top: 0.25rem;
  }
  .reason-list {
    list-style: none;
    margin-top: 0.75rem;
  }
  .reason-list li {
    padding: 0.5rem 0;
    border-bottom: 1px solid #243247;
    font-size: 0.95rem;
  }
  .reason-list li:last-child { border-bottom: none; }
  .trigger-box {
    margin-top: 1rem;
    background: rgba(239, 68, 68, 0.1);
    border-left: 4px solid var(--danger);
    padding: 0.85rem 1rem;
    border-radius: 0 6px 6px 0;
  }
</style>
</head>
<body>
<div class="container">
  <header>
    <h1>
      <svg width="34" height="34" viewBox="0 0 40 40" fill="none">
        <rect width="40" height="40" rx="8" fill="#0284c7"/>
        <path d="M10 12h20M10 20h20M10 28h12" stroke="#fff" stroke-width="3.5" stroke-linecap="round"/>
        <circle cx="28" cy="28" r="3.5" fill="#38bdf8"/>
      </svg>
      QueryLane
    </h1>
    <p>Cost-Aware LLM Query Routing & Lane Decision Explainer</p>
  </header>

  <div class="card">
    <div class="form-group">
      <label>Preset Prompt Examples</label>
      <div class="presets">
        <button class="preset-btn" onclick="setPreset('sentiment')">Sentiment (SST-2)</button>
        <button class="preset-btn" onclick="setPreset('ner')">Named Entities</button>
        <button class="preset-btn" onclick="setPreset('pal_math')">Compact PAL Math</button>
        <button class="preset-btn" onclick="setPreset('multi_math')">Multipart Math</button>
        <button class="preset-btn" onclick="setPreset('debug')">Code Debugging</button>
        <button class="preset-btn" onclick="setPreset('summary')">Summarization</button>
        <button class="preset-btn" onclick="setPreset('factual')">Ambiguous Factual</button>
      </div>
      <label for="prompt">Input Query / Prompt</label>
      <textarea id="prompt" placeholder="Enter a prompt to analyze..."></textarea>
    </div>
    <button class="btn-submit" onclick="analyze()">Explain Lane Decision</button>
  </div>

  <div id="results" class="card" style="display: none;">
    <h3 style="margin-bottom: 0.5rem;">Pipeline Decision</h3>

    <div class="lane-pipeline">
      <div id="lane-rules" class="lane-box">
        <div class="lane-title">1. Rules</div>
        <div class="lane-cost">0 Tokens</div>
      </div>
      <div id="lane-specialist" class="lane-box">
        <div class="lane-title">2. Specialist</div>
        <div class="lane-cost">0 Tokens (ONNX)</div>
      </div>
      <div id="lane-local" class="lane-box">
        <div class="lane-title">3. Local GGUF</div>
        <div class="lane-cost">0 Tokens (CPU)</div>
      </div>
      <div id="lane-cloud" class="lane-box">
        <div class="lane-title">4. Cloud</div>
        <div class="lane-cost">Billed API</div>
      </div>
    </div>

    <div class="stat-grid">
      <div class="stat-card">
        <label>Category</label>
        <div id="res-category" class="stat-val">-</div>
      </div>
      <div class="stat-card">
        <label>Confidence</label>
        <div id="res-confidence" class="stat-val">-</div>
      </div>
      <div class="stat-card">
        <label>Selected Lane</label>
        <div id="res-lane" class="stat-val">-</div>
      </div>
    </div>

    <div style="margin-bottom: 1rem;">
      <label>Estimated Token Cost</label>
      <div id="res-cost" style="font-weight: 600; color: #38bdf8; margin-top: 0.25rem;">-</div>
    </div>

    <label>Decision Rationale</label>
    <ul id="res-reasons" class="reason-list"></ul>

    <div id="triggers-wrap" class="trigger-box" style="display: none;">
      <strong>Escalation / Fallback Guards:</strong>
      <ul id="res-triggers" class="reason-list" style="margin-top: 0.25rem;"></ul>
    </div>
  </div>
</div>

<script>
const PRESETS = {
  sentiment: "Review: 'The battery lasts three days, build quality is solid, and the screen is gorgeous. Best laptop I\\'ve ever owned.'",
  ner: "Identify all organizations and dates mentioned in the text: Apple was founded by Steve Jobs in April 1976 in Cupertino.",
  pal_math: "A store has 120 shirts. 25% are sold on Monday, and 15 more are sold on Tuesday. How many shirts remain?",
  multi_math: "Alice has 14 apples and Bob has 9 oranges. How many apples does Alice have and how many oranges does Bob have?",
  debug: "def add_elements(lst):\\n    for i in range(len(lst)):\\n        lst[i] += 1\\n    return lst",
  summary: "Summarize the following passage in exactly three sentences: Quantum computing leverages quantum mechanical phenomena such as superposition and entanglement...",
  factual: "What is the capital of Australia?"
};

function setPreset(key) {
  document.getElementById('prompt').value = PRESETS[key] || '';
  analyze();
}

async function analyze() {
  const prompt = document.getElementById('prompt').value;
  if (!prompt.trim()) return;

  const resBox = document.getElementById('results');
  resBox.style.display = 'block';

  try {
    const resp = await fetch('/api/explain', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt })
    });
    const data = await resp.json();

    document.getElementById('res-category').textContent = data.category;
    document.getElementById('res-confidence').textContent = (data.confidence * 100).toFixed(0) + '%';

    const lane = data.predicted_lane.toLowerCase();
    const laneEl = document.getElementById('res-lane');
    laneEl.textContent = data.predicted_lane.toUpperCase();
    laneEl.className = 'stat-val badge badge-' + lane;

    document.getElementById('res-cost').textContent = data.estimated_token_cost;

    ['rules', 'specialist', 'local', 'cloud'].forEach(l => {
      document.getElementById('lane-' + l).classList.remove('active');
    });
    if (document.getElementById('lane-' + lane)) {
      document.getElementById('lane-' + lane).classList.add('active');
    }

    const reasonsList = document.getElementById('res-reasons');
    reasonsList.innerHTML = '';
    data.reasons.forEach(r => {
      const li = document.createElement('li');
      li.textContent = "• " + r;
      reasonsList.appendChild(li);
    });

    const triggerWrap = document.getElementById('triggers-wrap');
    const triggerList = document.getElementById('res-triggers');
    triggerList.innerHTML = '';
    if (data.escalation_triggers && data.escalation_triggers.length > 0) {
      triggerWrap.style.display = 'block';
      data.escalation_triggers.forEach(t => {
        const li = document.createElement('li');
        li.textContent = "⚠ " + t;
        triggerList.appendChild(li);
      });
    } else {
      triggerWrap.style.display = 'none';
    }
  } catch (err) {
    alert("Analysis failed: " + err);
  }
}
</script>
</body>
</html>
"""


class QueryLaneHandler(BaseHTTPRequestHandler):
    """Serve the static demo page and handle REST analysis requests."""

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            body = HTML_TEMPLATE.encode("utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif path == "/api/health":
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json")
            body = b'{"status":"ok","service":"querylane"}'
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_error(HTTPStatus.NOT_FOUND, "Not Found")

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/explain":
            length = int(self.headers.get("Content-Length", 0))
            raw_body = self.rfile.read(length).decode("utf-8")
            try:
                payload = json.loads(raw_body)
                prompt = payload.get("prompt", "")
                result = explain_decision(prompt)
                resp_bytes = json.dumps(result).encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(resp_bytes)))
                self.end_headers()
                self.wfile.write(resp_bytes)
            except Exception as exc:
                err_bytes = json.dumps({"error": str(exc)}).encode("utf-8")
                self.send_response(HTTPStatus.BAD_REQUEST)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(err_bytes)))
                self.end_headers()
                self.wfile.write(err_bytes)
        else:
            self.send_error(HTTPStatus.NOT_FOUND, "Not Found")

    def log_message(self, format: str, *args) -> None:
        # Keep demo terminal uncluttered
        pass


def run_server(host: str = "127.0.0.1", port: int = 8000) -> int:
    """Start the demonstration web server."""
    server_address = (host, port)
    server = ThreadingHTTPServer(server_address, QueryLaneHandler)
    print(f"QueryLane Web Demo running at http://{host}:{port}/ (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping web demo server.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 8000
    sys.exit(run_server(port=port))
