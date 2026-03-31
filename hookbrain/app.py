import json
import os
import subprocess
import sys
import tempfile
import threading
import uuid
from datetime import datetime, timezone

from flask import Flask, jsonify, render_template, request

from db import get_history, get_scan, init_db, save_scan

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT   = os.path.dirname(SCRIPT_DIR)
VENV_PYTHON = os.path.join(REPO_ROOT, "venv", "bin", "python")
SCANNER     = os.path.join(SCRIPT_DIR, "scanner.py")

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = Flask(__name__)

# In-memory job store  { job_id: {status, result, error} }
_jobs: dict = {}
_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Background scan worker
# ---------------------------------------------------------------------------
def _run_scan(job_id: str, hook_text: str, parent_scan_id=None, mechanic=None):
    tmp = tempfile.mktemp(suffix=".json")
    try:
        with _lock:
            _jobs[job_id]["status"] = "running"

        proc = subprocess.run(
            [VENV_PYTHON, SCANNER, hook_text, tmp],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
            timeout=900,
        )

        if proc.returncode != 0:
            stderr = (proc.stderr or "").strip()
            raise RuntimeError(stderr[-3000:] if stderr else "Scanner exited non-zero")

        with open(tmp) as f:
            data = json.load(f)

        scan_id = save_scan(hook_text, data,
                            parent_scan_id=parent_scan_id,
                            mechanic=mechanic)

        with _lock:
            _jobs[job_id]["status"] = "done"
            _jobs[job_id]["result"] = {**data, "scan_id": scan_id}

    except Exception as exc:
        with _lock:
            _jobs[job_id]["status"] = "error"
            _jobs[job_id]["error"]  = str(exc)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _start_job(hook_text: str, parent_scan_id=None, mechanic=None) -> str:
    job_id = str(uuid.uuid4())
    with _lock:
        _jobs[job_id] = {"status": "queued", "result": None, "error": None}
    threading.Thread(
        target=_run_scan,
        args=(job_id, hook_text, parent_scan_id, mechanic),
        daemon=True,
    ).start()
    return job_id


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/scan", methods=["POST"])
def api_scan():
    hook_text = (request.json or {}).get("hook", "").strip()
    if not hook_text:
        return jsonify({"error": "No hook text"}), 400
    job_id = _start_job(hook_text)
    return jsonify({"job_id": job_id})


@app.route("/api/scan/<job_id>")
def api_scan_status(job_id):
    with _lock:
        job = dict(_jobs.get(job_id) or {})
    if not job:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(job)


@app.route("/api/rewrites", methods=["POST"])
def api_rewrites():
    import anthropic

    body       = request.json or {}
    hook_text  = body.get("hook", "")
    brain_data = body.get("brain_data", {})
    scan_id    = body.get("scan_id")

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return jsonify({"error": "ANTHROPIC_API_KEY not set"}), 500

    prompt = f"""You are rewriting a short-form video hook. Your only job is to match a specific creator voice — spoken, immediate, never written.

══ VOICE RULES ══════════════════════════════════════════════════

DO:
- Start with "I", "You", a name, or a brand — never "Have you" or "Did you know" for top performers
- Use present tense and immediate verbs — "just", "now", "today"
- Make the viewer the subject or co-subject — "you watched", "everyone you know"
- Use sentence fragments and trailing off — creates curiosity without resolving
- Use specific numbers — "$20M", "1 billion views", "22,000 kiosks" — never vague scale
- Land contrast in a single line — "turned no brand into the biggest brand"
- Under 15 words is the sweet spot for the first spoken sentence

AVOID:
- Starting with a question — "Did you know", "Have you ever" — analytical, not emotional
- Grammatically complete sentences — sounds scripted
- Passive voice — "it was revealed that" vs "just revealed"
- Hedging — "might", "could", "perhaps"
- Generic openings — "In today's video", "So today", "Welcome back"
- Third-person distant framing — "A startup called Cluely" vs "You watched their videos"
- Numbers that require parsing — "billions of views and $20 million" makes the brain work

══ VOICE EXAMPLES (real hooks, verified view counts) ════════════

Study these. Match the register, rhythm, and compression:

  "Jason Derulo made $30 million washing cars."  — 2.1M views
  "I'm getting that itch again."  — 5.6M views
  "I never liked my smile."  — 3.7M views
  "You've never coded? Perfect!"  — 1.5M views
  "Wall Street has a fear meter, and it's called Vix."  — 2.1M views
  "You're not unmotivated, you've just been trained that effort won't matter."  — 1.1M views
  "No one is bored anymore and it's actually a problem."  — 1.1M views
  "A 2 MB zipfile turning into 75 GB is the closest thing computers have to black magic."  — 516K views
  "Everybody buys AI stocks, but nobody realizes..."  — 1.6M views
  "These two guys decided to not get a job after college."  — 1.2M views
  "So apparently we're going into a recession because girls are wearing maxi skirts."  — 2.9M views
  "$19 billion just vanished in crypto."  — 561K views
  "We are the most informed generation in history, yet somehow the dumbest."  — 827K views
  "Most people haven't caught on yet, but..."  — 814K views
  "How this Japanese brand turned no brand into the biggest brand."  — 1.9M views
  "I feel like people don't seem to realize that their opinions about the world is also a confession to their character."  — 2.3M views
  "Everyone is talking about MCP on the internet, so what the f*ck is MCP?"  — 1.3M views
  "China just deployed 22,000 AI powered health kiosks."  — 1.2M views

══ BRAIN DATA ═══════════════════════════════════════════════════

Original hook: "{hook_text}"

This hook was run through Meta's TRIBE v2 brain activation model (fMRI-trained, 20,484 cortical vertices).

Brain activation per second:
{json.dumps(brain_data.get("seconds", []), indent=2)}

Viral score breakdown:
{json.dumps(brain_data.get("viral", {}), indent=2)}

Brain rubric:
- WATCH signal (t0.mean + t1.mean high): viewer commits in first 2 seconds
- EMOTIONAL SALIENCE (top100_left_pct < 45% at t=0): right hemisphere fires = emotional hit
- SELF-RELEVANCE ("you/your" framing): medial prefrontal cortex = "this is about me"
- SHARE signal (top100_left_pct ~45-55%): bilateral = viewer thinks about others
- DROP-OFF WARNING (top100_left_pct > 55% at t=3): brain switched to analytical mode, viewer leaves

══ TASK ═════════════════════════════════════════════════════════

Write 5 rewrites for the same topic: Cluely — an app with 1 billion views, raised $20M, then shut down.

Each rewrite targets a different viral mechanic. Each must sound exactly like the voice examples above — spoken, not written, compressed, immediate.

Return ONLY a valid JSON array, no markdown fences, no text outside the JSON:
[
  {{"mechanic": "watch_signal",       "hook": "...", "why": "one sentence — which brain signal this targets and why the language achieves it"}},
  {{"mechanic": "self_relevance",     "hook": "...", "why": "..."}},
  {{"mechanic": "emotional_salience", "hook": "...", "why": "..."}},
  {{"mechanic": "share_signal",       "hook": "...", "why": "..."}},
  {{"mechanic": "dropoff_prevention", "hook": "...", "why": "..."}}
]"""

    client = anthropic.Anthropic(api_key=api_key)
    msg    = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )

    text = msg.content[0].text.strip()
    # Strip accidental markdown fences
    if "```" in text:
        parts = text.split("```")
        text = parts[1]
        if text.startswith("json"):
            text = text[4:]
    text = text.strip()

    rewrite_list = json.loads(text)
    return jsonify({"rewrites": rewrite_list})


@app.route("/api/scan_rewrites", methods=["POST"])
def api_scan_rewrites():
    body       = request.json or {}
    rewrites   = body.get("rewrites", [])
    parent_id  = body.get("scan_id")

    jobs_out = []
    for r in rewrites:
        job_id = _start_job(r["hook"],
                            parent_scan_id=parent_id,
                            mechanic=r.get("mechanic"))
        jobs_out.append({
            "job_id":   job_id,
            "mechanic": r.get("mechanic", ""),
            "hook":     r["hook"],
        })
    return jsonify({"jobs": jobs_out})


@app.route("/api/history")
def api_history():
    return jsonify({"history": get_history()})


@app.route("/api/history/<int:scan_id>")
def api_history_detail(scan_id):
    scan = get_scan(scan_id)
    if not scan:
        return jsonify({"error": "Not found"}), 404
    return jsonify(scan)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    init_db()
    app.run(host="127.0.0.1", port=5050, debug=False, use_reloader=False)
