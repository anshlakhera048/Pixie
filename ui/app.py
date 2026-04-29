"""Minimal web UI for Pixie — single-file FastAPI + HTML.

Run with:  python -m ui.app
Or:        uvicorn ui.app:app --reload

Provides:
  - Chat input/output
  - Live metrics panel
  - System status view
"""

from __future__ import annotations

import asyncio
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse

app = FastAPI(title="Pixie UI", docs_url=None, redoc_url=None)

# ------------------------------------------------------------------
# HTML Template (inline — no external dependencies)
# ------------------------------------------------------------------

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Pixie AI Assistant</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         background: #1a1a2e; color: #eee; height: 100vh; display: flex; }
  .sidebar { width: 280px; background: #16213e; padding: 1rem; overflow-y: auto;
             border-right: 1px solid #0f3460; }
  .sidebar h3 { color: #e94560; margin-bottom: 0.8rem; font-size: 0.85rem;
                text-transform: uppercase; letter-spacing: 1px; }
  .metric { background: #0f3460; border-radius: 6px; padding: 0.6rem;
            margin-bottom: 0.5rem; font-size: 0.8rem; }
  .metric .label { color: #aaa; }
  .metric .value { color: #4fc3f7; font-weight: bold; font-size: 1rem; }
  .main { flex: 1; display: flex; flex-direction: column; }
  .header { background: #16213e; padding: 1rem 1.5rem; border-bottom: 1px solid #0f3460;
            display: flex; align-items: center; gap: 1rem; }
  .header h1 { font-size: 1.3rem; color: #e94560; }
  .header .badge { background: #0f3460; color: #4fc3f7; padding: 0.2rem 0.6rem;
                   border-radius: 12px; font-size: 0.7rem; }
  .chat { flex: 1; overflow-y: auto; padding: 1.5rem; }
  .message { margin-bottom: 1rem; max-width: 80%; }
  .message.user { margin-left: auto; }
  .message .bubble { padding: 0.8rem 1rem; border-radius: 12px; line-height: 1.5;
                     font-size: 0.9rem; white-space: pre-wrap; }
  .message.user .bubble { background: #e94560; color: white; border-bottom-right-radius: 4px; }
  .message.assistant .bubble { background: #0f3460; border-bottom-left-radius: 4px; }
  .message .sender { font-size: 0.7rem; color: #888; margin-bottom: 0.2rem; }
  .input-area { padding: 1rem 1.5rem; background: #16213e; border-top: 1px solid #0f3460;
                display: flex; gap: 0.5rem; }
  .input-area input { flex: 1; background: #0f3460; border: 1px solid #1a1a2e;
                      color: #eee; padding: 0.8rem 1rem; border-radius: 8px;
                      font-size: 0.9rem; outline: none; }
  .input-area input:focus { border-color: #e94560; }
  .input-area button { background: #e94560; color: white; border: none;
                       padding: 0.8rem 1.5rem; border-radius: 8px; cursor: pointer;
                       font-weight: bold; font-size: 0.9rem; }
  .input-area button:hover { background: #d63851; }
  .input-area button:disabled { opacity: 0.5; cursor: not-allowed; }
  #metrics-panel .metric { transition: all 0.3s; }
</style>
</head>
<body>
<div class="sidebar" id="metrics-panel">
  <h3>System Metrics</h3>
  <div class="metric"><span class="label">Status</span><div class="value" id="m-status">Loading...</div></div>
  <div class="metric"><span class="label">Uptime</span><div class="value" id="m-uptime">-</div></div>
  <div class="metric"><span class="label">Requests</span><div class="value" id="m-requests">0</div></div>
  <div class="metric"><span class="label">Error Rate</span><div class="value" id="m-errors">0%</div></div>
  <h3 style="margin-top:1rem;">Cache</h3>
  <div class="metric"><span class="label">LLM Hit Rate</span><div class="value" id="m-cache-llm">-</div></div>
  <div class="metric"><span class="label">Tool Hit Rate</span><div class="value" id="m-cache-tool">-</div></div>
  <h3 style="margin-top:1rem;">Tools</h3>
  <div id="m-tools" style="font-size:0.8rem; color:#aaa;">No data</div>
</div>
<div class="main">
  <div class="header">
    <h1>Pixie</h1>
    <span class="badge">AI Assistant</span>
  </div>
  <div class="chat" id="chat"></div>
  <div class="input-area">
    <input type="text" id="input" placeholder="Ask Pixie anything..." autocomplete="off" />
    <button id="send-btn" onclick="sendMessage()">Send</button>
  </div>
</div>
<script>
const chat = document.getElementById('chat');
const input = document.getElementById('input');
const btn = document.getElementById('send-btn');

input.addEventListener('keydown', e => { if (e.key === 'Enter' && !btn.disabled) sendMessage(); });

async function sendMessage() {
  const text = input.value.trim();
  if (!text) return;
  input.value = '';
  btn.disabled = true;
  appendMessage('user', text);

  try {
    const res = await fetch('/ui/chat', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({message: text})
    });
    const data = await res.json();
    appendMessage('assistant', data.response || data.error || 'No response');
  } catch (err) {
    appendMessage('assistant', 'Connection error: ' + err.message);
  }
  btn.disabled = false;
  input.focus();
}

function appendMessage(role, text) {
  const div = document.createElement('div');
  div.className = 'message ' + role;
  div.innerHTML = `<div class="sender">${role === 'user' ? 'You' : 'Pixie'}</div><div class="bubble">${escapeHtml(text)}</div>`;
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
}

function escapeHtml(t) {
  return t.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

async function refreshMetrics() {
  try {
    const res = await fetch('/ui/metrics');
    const d = await res.json();
    document.getElementById('m-status').textContent = 'Online';
    document.getElementById('m-uptime').textContent = Math.round(d.uptime_seconds || 0) + 's';
    document.getElementById('m-requests').textContent = d.total_requests || 0;
    const errRate = d.error_rate != null ? (d.error_rate * 100).toFixed(1) + '%' : '0%';
    document.getElementById('m-errors').textContent = errRate;
    if (d.cache) {
      const llmRate = d.cache.llm ? (d.cache.llm.hit_rate * 100).toFixed(0) + '%' : '-';
      const toolRate = d.cache.tools ? (d.cache.tools.hit_rate * 100).toFixed(0) + '%' : '-';
      document.getElementById('m-cache-llm').textContent = llmRate;
      document.getElementById('m-cache-tool').textContent = toolRate;
    }
  } catch(e) {}
}

setInterval(refreshMetrics, 5000);
refreshMetrics();
</script>
</body>
</html>"""


# ------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML_PAGE


@app.post("/ui/chat")
async def ui_chat(request: Request):
    """Process a chat message through the agent."""
    body = await request.json()
    message = body.get("message", "").strip()
    if not message:
        return JSONResponse({"error": "Empty message"}, status_code=400)

    try:
        from core.orchestrator import Orchestrator

        orchestrator = _get_orchestrator()
        # Collect streamed response
        chunks: list[str] = []
        async for chunk in orchestrator.runtime.submit_streaming(message):
            chunks.append(chunk)
        response = "".join(chunks)
        return {"response": response}
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


@app.get("/ui/metrics")
async def ui_metrics():
    """Return current system metrics for the dashboard."""
    try:
        from observability.metrics import get_metrics
        return get_metrics().snapshot()
    except Exception:
        return {"uptime_seconds": 0, "error": "metrics unavailable"}


# ------------------------------------------------------------------
# Lazy orchestrator singleton
# ------------------------------------------------------------------

_orchestrator = None


def _get_orchestrator():
    """Lazy-init the orchestrator for UI mode."""
    global _orchestrator
    if _orchestrator is None:
        from core.orchestrator import Orchestrator
        _orchestrator = Orchestrator()
    return _orchestrator


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PIXIE_UI_PORT", "8501"))
    print(f"  Pixie UI starting on http://localhost:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")
