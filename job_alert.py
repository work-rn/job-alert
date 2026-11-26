"""
Daily AI-powered job alert:
1. Fetch recent Malaysia LinkedIn jobs + full descriptions via Apify (`geoId=106808692`).
2. Use Residential proxies so LinkedIn does not block detail pages (`desc_len > 0`).
3. Batch-evaluate all candidates in ONE Gemini API call (stays well under 5 RPM Free Tier quota)
   with automatic retry & fallback on 503/429 errors.
4. Update `seen_jobs.json`, generate an interactive GitHub Pages dashboard (`docs/index.html`) with live
   scan configuration controls, and email the new matches.
"""

import json
import os
import re
import smtplib
import ssl
import time
from datetime import datetime, timezone
from email.message import EmailMessage
from html import escape, unescape
from pathlib import Path
from urllib.parse import quote_plus

from apify_client import ApifyClient
from google import genai
from pydantic import BaseModel, Field

# ---- Settings (Overridable via Webpage UI, Workflow Inputs, or Repo Vars/Secrets) ----
ACTOR_ID = "mighty_monk/linkedin-jobs-scraper"
MALAYSIA_GEO_ID = "106808692"

MALAYSIA_TERMS = (
    "malaysia",
    "kuala lumpur",
    "selangor",
    "penang",
    "pulau pinang",
    "johor",
    "cyberjaya",
    "putrajaya",
    "petaling",
    "shah alam",
    "subang",
    "puchong",
    "klang",
    "damansara",
    "bangsar",
    "mont kiara",
    "perak",
    "ipoh",
    "melaka",
    "malacca",
    "negeri sembilan",
    "seremban",
    "pahang",
    "kuantan",
    "kedah",
    "sarawak",
    "kuching",
    "sabah",
    "kota kinabalu",
)

DEFAULT_QUERIES = [
    '("IT Service" OR "Service Delivery" OR "IT Operations" OR "Incident Manager" OR "Problem Manager" OR "Change Manager") AND ("Manager" OR "Lead" OR "Owner") NOT "Junior" NOT "Intern"',
    '("CMDB" OR "ITAM" OR "Asset Management" OR "Configuration Manager" OR "BMC Discovery" OR "ServiceNow" OR "AI" OR "GenAI" OR "Digital Transformation") AND ("Manager" OR "Lead" OR "Owner" OR "Delivery") NOT "Junior"',
]


def _env(name: str, default: str) -> str:
    value = os.environ.get(name, "").strip()
    return value or default


def _env_list(name: str, default: list[str]) -> list[str]:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    parts = re.split(r"[\r\n]+|\s*;\s*", raw)
    if len(parts) == 1 and "," in raw and '"' not in raw:
        parts = raw.split(",")
    return [p.strip() for p in parts if p.strip()]


QUERIES = _env_list("SEARCH_QUERIES", DEFAULT_QUERIES)
LOCATION = _env("SEARCH_LOCATION", "Malaysia")
MAX_JOBS = int(_env("MAX_JOBS", "15"))
DEFAULT_HOURS_BACK = int(_env("HOURS_BACK", "24"))
MIN_MATCH_SCORE = int(_env("MIN_MATCH_SCORE", "7"))
IGNORE_SEEN = _env("IGNORE_SEEN", "false").lower() in ("1", "true", "yes")
GEMINI_MODEL = _env("GEMINI_MODEL", "gemini-3.8-flash")
FALLBACK_MODELS = ["gemini-3.8-flash", "gemini-3.5-flash-lite"]

STATE_FILE = Path("seen_jobs.json")
DASHBOARD_FILE = Path("docs/index.html")
# ---------------------------------------------------------------------------


class SingleJobEvaluation(BaseModel):
    job_id: str = Field(description="Exact job_id string from the input job list")
    score: int = Field(description="Match score from 1 to 10 against candidate CV")
    is_match: bool = Field(description="True if score >= MIN_MATCH_SCORE and seniority/domain fit")
    reason: str = Field(description="1-2 concise sentences on why this fits or key gaps")
    tailored_bullets: list[str] = Field(
        description="3 tailored resume bullet points aligned to this job (empty list if score < MIN_MATCH_SCORE)"
    )


class BatchJobEvaluation(BaseModel):
    evaluations: list[SingleJobEvaluation]


def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"last_run_utc": None, "seen": {}, "matched_jobs": []}


def save_state(state: dict) -> None:
    seen_items = list(state.get("seen", {}).items())
    if len(seen_items) > 1500:
        state["seen"] = dict(seen_items[-1500:])
    state["matched_jobs"] = state.get("matched_jobs", [])[:100]
    STATE_FILE.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")


def compute_hours_back() -> int:
    return DEFAULT_HOURS_BACK


def search_url(query: str, hours_back: int) -> str:
    geo_param = f"&geoId={MALAYSIA_GEO_ID}" if LOCATION.strip().lower() == "malaysia" else ""
    return (
        "https://www.linkedin.com/jobs/search/"
        f"?keywords={quote_plus(query)}&location={quote_plus(LOCATION)}"
        f"{geo_param}&f_TPR=r{hours_back * 3600}&sortBy=DD"
    )


def _is_target_location(location_str: str) -> bool:
    if LOCATION.strip().lower() != "malaysia":
        return True
    loc = (location_str or "").strip().lower()
    if not loc:
        return True
    return any(term in loc for term in MALAYSIA_TERMS)


def _extract_description(job: dict) -> str:
    for key in ("description", "descriptionText", "jobDescription", "text"):
        val = job.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    html_val = job.get("descriptionHtml") or job.get("jobDescriptionHtml") or ""
    if isinstance(html_val, str) and html_val.strip():
        clean = re.sub(r"<[^>]+>", " ", html_val)
        return re.sub(r"\s+", " ", unescape(clean)).strip()
    return ""


def fetch_new_jobs(seen_ids: set[str], hours_back: int) -> list[dict]:
    client = ApifyClient(os.environ["APIFY_TOKEN"])
    urls = [search_url(q, hours_back) for q in QUERIES]

    print(
        f"Scanning {len(urls)} search URL(s) in '{LOCATION}' for last {hours_back}h "
        f"(ignore_seen={IGNORE_SEEN}, max_total={MAX_JOBS}, min_score={MIN_MATCH_SCORE}, proxy=RESIDENTIAL)..."
    )
    run_input = {
        "searchUrls": urls,
        "maxJobs": MAX_JOBS,
        "maxConcurrency": 1,
        "requestDelayMs": 3000,
        "maxRequestsPerMinute": 15,
        "maxRequestRetries": 5,
        "includeDescription": True,
        "proxyConfiguration": {
            "useApifyProxy": True,
            "apifyProxyGroups": ["RESIDENTIAL"],
        },
    }
    run = client.actor(ACTOR_ID).call(run_input=run_input)
    dataset_id = run["defaultDatasetId"] if isinstance(run, dict) else run.default_dataset_id
    items = client.dataset(dataset_id).list_items().items

    new_candidates = []
    batch_seen = set()
    skipped_seen = 0
    skipped_location = 0

    for item in items:
        if len(new_candidates) >= MAX_JOBS:
            break

        job_id = str(item.get("jobId") or item.get("id") or item.get("jobUrl") or "").strip()
        job_url = (item.get("jobUrl") or item.get("applyUrl") or "").strip()
        if not job_id or not job_url or not item.get("title"):
            continue
        if job_id in batch_seen:
            continue
        if not IGNORE_SEEN and job_id in seen_ids:
            skipped_seen += 1
            continue

        loc_raw = item.get("location") or ""
        if not _is_target_location(loc_raw):
            skipped_location += 1
            continue

        batch_seen.add(job_id)
        item["_clean_id"] = job_id
        item["_clean_url"] = job_url
        new_candidates.append(item)

    print(
        f"Apify returned {len(items)} total -> skipped {skipped_seen} already-seen, "
        f"{skipped_location} outside '{LOCATION}' -> evaluating {len(new_candidates)} job(s) (capped at {MAX_JOBS})."
    )
    return new_candidates


def _discover_available_models(ai_client: genai.Client) -> list[str]:
    discovered = []
    try:
        for m in ai_client.models.list():
            name = getattr(m, "name", "") or ""
            short = name.replace("models/", "")
            methods = getattr(m, "supported_generation_methods", None) or getattr(m, "supported_actions", None) or []
            if "gemini" in short and ("flash" in short or "pro" in short) and "embed" not in short and "image" not in short and "audio" not in short:
                if not methods or "generateContent" in methods:
                    discovered.append(short)
    except Exception:
        pass
    return discovered


def _call_gemini_with_retry(ai_client: genai.Client, prompt: str) -> dict:
    models_to_try = [GEMINI_MODEL] + [m for m in FALLBACK_MODELS if m != GEMINI_MODEL]
    for extra in _discover_available_models(ai_client):
        if extra not in models_to_try:
            models_to_try.append(extra)
    last_err = None

    for model_name in models_to_try:
        for attempt in range(1, 4):
            try:
                print(f"Calling Gemini ({model_name}, attempt {attempt}/3)...")
                response = ai_client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config={
                        "response_mime_type": "application/json",
                        "response_schema": BatchJobEvaluation,
                        "temperature": 0.2,
                    },
                )
                return json.loads(response.text)
            except Exception as exc:
                last_err = exc
                err_str = str(exc)
                print(f"  -> {model_name} attempt {attempt} failed: {err_str[:220]}")
                if "404" in err_str:
                    break  # Try next model immediately
                wait_sec = 12 * attempt
                print(f"  -> Waiting {wait_sec}s before retry...")
                time.sleep(wait_sec)

    raise RuntimeError(f"All Gemini models failed after retries. Last error: {last_err}")


def evaluate_jobs_with_ai(jobs: list[dict], cv_text: str, seen_map: dict) -> list[dict]:
    if not jobs:
        return []

    ai_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    job_map = {}
    job_blocks = []
    for job in jobs:
        job_id = job["_clean_id"]
        title = job.get("title", "Unknown Role")
        company = job.get("company") or job.get("companyName") or "Unknown Company"
        loc = job.get("location") or LOCATION
        desc = _extract_description(job)

        print(f"Queued [{job_id}]: {title} @ {company} ({loc}) | desc_len={len(desc)}")
        job_map[job_id] = {
            "job": job,
            "title": title,
            "company": company,
            "loc": loc,
            "has_desc": bool(desc),
        }
        desc_text = desc[:1500] if desc else "[Unavailable due to LinkedIn anti-bot wall — evaluate strictly from Job Title, Company, Seniority, and Location.]"
        job_blocks.append(
            f"""---
JOB_ID: {job_id}
TITLE: {title}
COMPANY: {company}
LOCATION: {loc}
DESCRIPTION: {desc_text}"""
        )

    prompt = f"""
Evaluate how well this candidate matches EACH of the {len(job_blocks)} job postings below.
Return an evaluation entry for EVERY single JOB_ID provided.

Scoring rules (be realistic and selective):
- Score >= {MIN_MATCH_SCORE} ONLY if the candidate's seniority, leadership scope, and core technical/domain experience genuinely align (e.g., ITSM, Service Delivery, Incident/Problem/Change Management, CMDB/Asset Discovery, ServiceNow, IT Operations, AI/Automation Service Excellence).
- Score <= 5 for junior roles, pure sales/pre-sales quotas, language-specialist roles requiring Japanese/Korean fluency not in the CV, or unrelated domains (e.g., retail banking branch/division manager).
- Only generate 3 tailored, metric-oriented resume bullet points for jobs scoring >= {MIN_MATCH_SCORE} (use [] for lower-scoring jobs to keep output concise).

CANDIDATE CV:
{cv_text[:7000]}

JOB LISTINGS TO EVALUATE:
{chr(10).join(job_blocks)}
"""

    batch_result = _call_gemini_with_retry(ai_client, prompt)
    evaluations = batch_result.get("evaluations", [])
    matched = []

    for ev in evaluations:
        job_id = str(ev.get("job_id", "")).strip()
        if job_id not in job_map:
            continue
        meta = job_map[job_id]
        job = meta["job"]
        title = meta["title"]
        company = meta["company"]
        loc = meta["loc"]
        fallback_note = "" if meta["has_desc"] else " (Title-based evaluation — click link to view full description on LinkedIn)"

        score = int(ev.get("score", 0))
        is_match = bool(ev.get("is_match")) and score >= MIN_MATCH_SCORE
        print(f"  [{job_id}] {title} @ {company} -> AI Score: {score}/10 | Match: {is_match}")

        if is_match:
            seen_map[job_id] = {"status": f"matched_{score}", "date": today_str, "title": title}
            matched.append({
                "id": job_id,
                "title": title,
                "company": company,
                "location": loc,
                "posted_at": str(job.get("postedAt") or today_str),
                "found_date": today_str,
                "url": job["_clean_url"],
                "score": score,
                "reason": (ev.get("reason", "") + fallback_note).strip(),
                "bullets": ev.get("tailored_bullets", []),
            })
        else:
            seen_map[job_id] = {"status": f"rejected_{score}", "date": today_str, "title": title}

    matched.sort(key=lambda x: x["score"], reverse=True)
    return matched


def build_email(new_matches: list[dict], hours_back: int) -> tuple[str, str]:
    today = datetime.now().strftime("%d %b %Y")
    subject = f"AI Job Radar ({today}): {len(new_matches)} high-match role(s)"

    repo = os.environ.get("GITHUB_REPOSITORY", "")
    pages_link = ""
    if "/" in repo:
        owner, name = repo.split("/", 1)
        pages_url = f"https://{owner}.github.io/{name}/"
        pages_link = f'<p><a href="{pages_url}" style="display:inline-block;padding:10px 16px;background:#0f172a;color:#ffffff;text-decoration:none;border-radius:6px;font-weight:bold;">Open Live Job Dashboard &rarr;</a></p>'

    if not new_matches:
        body = (
            f"<p>No {escape(LOCATION)} jobs scored &ge; {MIN_MATCH_SCORE}/10 in the last {hours_back} hours.</p>"
            f"{pages_link}"
        )
        return subject, body

    cards = []
    for j in new_matches:
        bullets_html = "".join(f"<li>{escape(b)}</li>" for b in j.get("bullets", []))
        cards.append(
            f'<div style="border:1px solid #e2e8f0;border-radius:8px;padding:14px;margin-bottom:14px;">'
            f'<div style="font-size:16px;font-weight:bold;">'
            f'<span style="background:#dcfce7;color:#166534;padding:2px 8px;border-radius:12px;font-size:12px;margin-right:8px;">'
            f'{j["score"]}/10 Match</span>'
            f'<a href="{escape(j["url"])}" style="color:#0369a1;text-decoration:none;">{escape(j["title"])}</a>'
            f'</div>'
            f'<div style="color:#475569;font-size:13px;margin:4px 0 8px;">'
            f'{escape(j["company"])} &middot; {escape(j["location"])}</div>'
            f'<p style="margin:6px 0;font-size:14px;"><b>AI Fit Note:</b> {escape(j["reason"])}</p>'
            f'<div style="font-size:13px;color:#334155;"><b>Suggested CV Bullets:</b>'
            f'<ul style="margin:4px 0 0 18px;">{bullets_html}</ul></div>'
            f'</div>'
        )

    body = (
        f"<p>Found <b>{len(new_matches)}</b> role(s) in {escape(LOCATION)} matching your CV (&ge; {MIN_MATCH_SCORE}/10):</p>"
        f"{pages_link}"
        f'{"".join(cards)}'
    )
    return subject, body


def send_email(subject: str, html_body: str) -> None:
    sender = os.environ["GMAIL_ADDRESS"].strip()
    recipient = os.environ.get("EMAIL_TO", "").strip() or sender

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = recipient
    msg.set_content("Your email client does not support HTML.")
    msg.add_alternative(html_body, subtype="html")

    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=ssl.create_default_context()) as smtp:
        smtp.login(sender, os.environ["GMAIL_APP_PASSWORD"].strip())
        smtp.send_message(msg)


def generate_dashboard(all_matches: list[dict], last_run_utc: str) -> None:
    DASHBOARD_FILE.parent.mkdir(parents=True, exist_ok=True)
    repo = os.environ.get("GITHUB_REPOSITORY", "work-rn/job-alert")
    owner, repo_name = repo.split("/", 1) if "/" in repo else ("work-rn", "job-alert")
    jobs_json = json.dumps(all_matches, ensure_ascii=False)
    default_queries_str = "\n".join(QUERIES)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>AI Job Radar Dashboard</title>
  <style>
    :root {{ --bg: #f8fafc; --card: #ffffff; --text: #0f172a; --muted: #64748b; --border: #e2e8f0; --accent: #0f172a; }}
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: var(--bg); color: var(--text); margin: 0; padding: 24px 16px; }}
    .container {{ max-width: 920px; margin: 0 auto; }}
    header {{ display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center; gap: 12px; margin-bottom: 16px; }}
    h1 {{ margin: 0; font-size: 24px; }}
    .sub {{ color: var(--muted); font-size: 13px; margin-top: 4px; }}
    .controls {{ display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 18px; }}
    input, select, button, textarea {{ padding: 9px 12px; border: 1px solid var(--border); border-radius: 8px; font-size: 14px; background: #fff; font-family: inherit; }}
    input[type="text"], input[type="number"] {{ flex: 1; min-width: 140px; }}
    textarea {{ width: 100%; box-sizing: border-box; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; }}
    button.primary {{ background: var(--accent); color: #fff; border: none; font-weight: 600; cursor: pointer; }}
    button.primary:disabled {{ opacity: 0.6; cursor: not-allowed; }}
    button.secondary {{ background: #fff; color: var(--text); cursor: pointer; }}
    #statusBanner {{ display: none; padding: 12px 14px; border-radius: 8px; background: #eff6ff; border: 1px solid #bfdbfe; color: #1e40af; font-size: 14px; margin-bottom: 16px; }}
    .settings-panel {{ display: none; background: #fff; border: 1px solid var(--border); border-radius: 10px; padding: 16px; margin-bottom: 18px; box-shadow: 0 1px 3px rgba(0,0,0,0.04); }}
    .settings-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin-bottom: 12px; }}
    .field label {{ display: block; font-size: 12px; font-weight: 600; color: #334155; margin-bottom: 4px; }}
    .field input, .field select {{ width: 100%; box-sizing: border-box; }}
    .checkbox-row {{ display: flex; align-items: center; gap: 8px; font-size: 13px; font-weight: 600; color: #334155; margin-top: 22px; }}
    .checkbox-row input {{ width: 16px; height: 16px; margin: 0; }}
    .card {{ background: var(--card); border: 1px solid var(--border); border-radius: 10px; padding: 18px; margin-bottom: 14px; box-shadow: 0 1px 2px rgba(0,0,0,0.03); }}
    .card-top {{ display: flex; justify-content: space-between; align-items: flex-start; gap: 12px; }}
    .badge {{ background: #dcfce7; color: #166534; font-weight: 700; font-size: 12px; padding: 4px 10px; border-radius: 999px; white-space: nowrap; }}
    .badge.mid {{ background: #fef9c3; color: #854d0e; }}
    .title {{ font-size: 18px; font-weight: 700; color: #0369a1; text-decoration: none; }}
    .meta {{ color: var(--muted); font-size: 13px; margin: 4px 0 10px; }}
    .reason {{ font-size: 14px; line-height: 1.5; margin: 0 0 10px; }}
    details summary {{ cursor: pointer; font-size: 13px; font-weight: 600; color: #334155; }}
    details ul {{ margin: 8px 0 0 18px; font-size: 13px; color: #334155; line-height: 1.5; }}
    .apply-btn {{ display: inline-block; margin-top: 12px; padding: 7px 14px; background: #0369a1; color: #fff; text-decoration: none; border-radius: 6px; font-size: 13px; font-weight: 600; }}
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div>
        <h1>AI Job Radar</h1>
        <div class="sub">Last updated: {escape(last_run_utc)} &middot; <span id="countLabel">0</span> saved matches</div>
      </div>
      <div style="display:flex; gap:8px; flex-wrap:wrap;">
        <button class="secondary" onclick="toggleSettings()">&#9881; Scan Settings</button>
        <button class="secondary" onclick="resetToken()" title="Update GitHub Token">API Key</button>
        <button id="runBtn" class="primary" onclick="triggerScan()">&#9654; Run Scan Now</button>
      </div>
    </header>

    <div id="settingsPanel" class="settings-panel">
      <div class="settings-grid">
        <div class="field">
          <label for="cfgHoursBack">Time Window (Hours Back)</label>
          <select id="cfgHoursBack" onchange="saveConfig()">
            <option value="24">24 Hours (Daily)</option>
            <option value="72">72 Hours (3 Days)</option>
            <option value="168">168 Hours (7 Days)</option>
          </select>
        </div>
        <div class="field">
          <label for="cfgMaxJobs">Max Jobs to Check</label>
          <input type="number" id="cfgMaxJobs" min="1" max="50" value="{MAX_JOBS}" onchange="saveConfig()" />
        </div>
        <div class="field">
          <label for="cfgMinScore">Min AI Match Score (1-10)</label>
          <input type="number" id="cfgMinScore" min="1" max="10" value="{MIN_MATCH_SCORE}" onchange="saveConfig()" />
        </div>
        <div class="field">
          <label for="cfgLocation">Location</label>
          <input type="text" id="cfgLocation" value="{escape(LOCATION)}" onchange="saveConfig()" />
        </div>
        <div class="field">
          <label class="checkbox-row">
            <input type="checkbox" id="cfgIgnoreSeen" onchange="saveConfig()" />
            Re-check already seen jobs (ignore_seen)
          </label>
        </div>
      </div>
      <div class="field">
        <label for="cfgQueries">Search Queries (one per line)</label>
        <textarea id="cfgQueries" rows="3" onchange="saveConfig()">{escape(default_queries_str)}</textarea>
      </div>
      <div style="display:flex; justify-content:space-between; align-items:center; margin-top:10px;">
        <span style="font-size:12px; color:#64748b;">Settings are saved in your browser and sent when you click <b>Run Scan Now</b>.</span>
        <button class="secondary" style="padding:5px 10px; font-size:12px;" onclick="resetConfig()">Reset Defaults</button>
      </div>
    </div>

    <div id="statusBanner"></div>

    <div class="controls">
      <input type="text" id="searchInput" placeholder="Filter by title, company, or keyword..." oninput="renderJobs()" />
      <select id="scoreFilter" onchange="renderJobs()">
        <option value="0">All Saved Scores</option>
        <option value="9">9+ / 10 Only</option>
        <option value="8">8+ / 10 Only</option>
        <option value="7">7+ / 10 Only</option>
      </select>
    </div>

    <div id="jobList"></div>
  </div>

  <script>
    const REPO_OWNER = "{escape(owner)}";
    const REPO_NAME = "{escape(repo_name)}";
    const WORKFLOW_FILE = "job_alert.yml";
    const JOBS = {jobs_json};

    const SERVER_DEFAULTS = {{
      hours_back: "{DEFAULT_HOURS_BACK}",
      ignore_seen: {str(IGNORE_SEEN).lower()},
      max_jobs: "{MAX_JOBS}",
      min_match_score: "{MIN_MATCH_SCORE}",
      search_location: {json.dumps(LOCATION)},
      search_queries: {json.dumps(default_queries_str)}
    }};

    function loadConfig() {{
      let saved = {{}};
      try {{
        saved = JSON.parse(localStorage.getItem("gh_job_radar_cfg") || "{{}}");
      }} catch (e) {{}}
      const cfg = Object.assign({{}}, SERVER_DEFAULTS, saved);
      document.getElementById("cfgHoursBack").value = String(cfg.hours_back);
      document.getElementById("cfgIgnoreSeen").checked = Boolean(cfg.ignore_seen);
      document.getElementById("cfgMaxJobs").value = String(cfg.max_jobs);
      document.getElementById("cfgMinScore").value = String(cfg.min_match_score);
      document.getElementById("cfgLocation").value = String(cfg.search_location);
      document.getElementById("cfgQueries").value = String(cfg.search_queries);
      return cfg;
    }}

    function saveConfig() {{
      const cfg = {{
        hours_back: document.getElementById("cfgHoursBack").value,
        ignore_seen: document.getElementById("cfgIgnoreSeen").checked,
        max_jobs: document.getElementById("cfgMaxJobs").value,
        min_match_score: document.getElementById("cfgMinScore").value,
        search_location: document.getElementById("cfgLocation").value.trim() || "Malaysia",
        search_queries: document.getElementById("cfgQueries").value.trim()
      }};
      localStorage.setItem("gh_job_radar_cfg", JSON.stringify(cfg));
      return cfg;
    }}

    function resetConfig() {{
      localStorage.removeItem("gh_job_radar_cfg");
      loadConfig();
    }}

    function toggleSettings() {{
      const panel = document.getElementById("settingsPanel");
      panel.style.display = panel.style.display === "block" ? "none" : "block";
    }}

    function escapeHtml(str) {{
      return String(str || "").replace(/[&<>"']/g, m => ({{
        "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
      }}[m]));
    }}

    function renderJobs() {{
      const q = document.getElementById("searchInput").value.toLowerCase();
      const minScore = parseInt(document.getElementById("scoreFilter").value, 10);
      const filtered = JOBS.filter(j => {{
        const hay = `${{j.title}} ${{j.company}} ${{j.location}} ${{j.reason}}`.toLowerCase();
        return j.score >= minScore && hay.includes(q);
      }});

      document.getElementById("countLabel").textContent = filtered.length;
      const list = document.getElementById("jobList");
      if (!filtered.length) {{
        list.innerHTML = '<div class="card">No matching jobs found for the selected filter.</div>';
        return;
      }}

      list.innerHTML = filtered.map(j => `
        <div class="card">
          <div class="card-top">
            <div>
              <a class="title" href="${{escapeHtml(j.url)}}" target="_blank" rel="noopener">${{escapeHtml(j.title)}}</a>
              <div class="meta">${{escapeHtml(j.company)}} &middot; ${{escapeHtml(j.location)}} &middot; Found ${{escapeHtml(j.found_date)}}</div>
            </div>
            <span class="badge ${{j.score < 8 ? 'mid' : ''}}">${{j.score}}/10 Match</span>
          </div>
          <p class="reason"><strong>AI Fit Note:</strong> ${{escapeHtml(j.reason)}}</p>
          <details>
            <summary>View 3 Tailored CV Bullet Points</summary>
            <ul>${{(j.bullets || []).map(b => `<li>${{escapeHtml(b)}}</li>`).join("")}}</ul>
          </details>
          <a class="apply-btn" href="${{escapeHtml(j.url)}}" target="_blank" rel="noopener">View &amp; Apply on LinkedIn &rarr;</a>
        </div>
      `).join("");
    }}

    function getToken() {{
      let token = localStorage.getItem("gh_job_radar_pat");
      if (!token) {{
        token = prompt("Enter your Fine-Grained GitHub PAT (stored only in this browser's localStorage):");
        if (token && token.trim()) {{
          localStorage.setItem("gh_job_radar_pat", token.trim());
        }}
      }}
      return token ? token.trim() : null;
    }}

    function resetToken() {{
      localStorage.removeItem("gh_job_radar_pat");
      alert("Saved GitHub token cleared from this browser.");
    }}

    function setBanner(msg) {{
      const el = document.getElementById("statusBanner");
      el.style.display = "block";
      el.textContent = msg;
    }}

    async function triggerScan() {{
      const token = getToken();
      if (!token) return;
      const cfg = saveConfig();

      const btn = document.getElementById("runBtn");
      btn.disabled = true;
      setBanner(`Triggering scan (${{cfg.hours_back}}h window, max ${{cfg.max_jobs}} jobs, ignore_seen=${{cfg.ignore_seen}})...`);

      try {{
        const dispatchRes = await fetch(
          `https://api.github.com/repos/${{REPO_OWNER}}/${{REPO_NAME}}/actions/workflows/${{WORKFLOW_FILE}}/dispatches`,
          {{
            method: "POST",
            headers: {{
              "Authorization": `Bearer ${{token}}`,
              "Accept": "application/vnd.github+json",
              "Content-Type": "application/json"
            }},
            body: JSON.stringify({{
              ref: "main",
              inputs: {{
                hours_back: String(cfg.hours_back),
                ignore_seen: String(cfg.ignore_seen),
                max_jobs: String(cfg.max_jobs),
                min_match_score: String(cfg.min_match_score),
                search_location: String(cfg.search_location),
                search_queries: String(cfg.search_queries)
              }}
            }})
          }}
        );

        if (!dispatchRes.ok) {{
          throw new Error(`GitHub API error (${{dispatchRes.status}}). Check repo name, PAT permissions, and workflow inputs.`);
        }}

        setBanner("Workflow queued! Polling status every 10s (usually takes 60-90s)...");
        await new Promise(r => setTimeout(r, 6000));

        for (let i = 0; i < 24; i++) {{
          const runsRes = await fetch(
            `https://api.github.com/repos/${{REPO_OWNER}}/${{REPO_NAME}}/actions/workflows/${{WORKFLOW_FILE}}/runs?per_page=1`,
            {{ headers: {{ "Authorization": `Bearer ${{token}}`, "Accept": "application/vnd.github+json" }} }}
          );
          const data = await runsRes.json();
          const latest = data.workflow_runs && data.workflow_runs[0];
          if (latest) {{
            setBanner(`Workflow status: ${{latest.status}} ${{latest.conclusion ? '(' + latest.conclusion + ')' : ''}}...`);
            if (latest.status === "completed") {{
              setBanner("Scan complete! Waiting 18s for GitHub Pages to publish, then refreshing...");
              await new Promise(r => setTimeout(r, 18000));
              location.reload();
              return;
            }}
          }}
          await new Promise(r => setTimeout(r, 10000));
        }}
        setBanner("Workflow is still running in background. Refresh this page shortly.");
      }} catch (err) {{
        setBanner("Error: " + err.message);
      }} finally {{
        btn.disabled = false;
      }}
    }}

    loadConfig();
    renderJobs();
  </script>
</body>
</html>
"""
    DASHBOARD_FILE.write_text(html, encoding="utf-8")


def main() -> None:
    cv_text = os.environ.get("CV_TEXT", "").strip()
    if not cv_text:
        raise ValueError("CV_TEXT environment variable/secret is missing. Add your CV to GitHub Secrets as CV_TEXT.")

    state = load_state()
    seen_map = state.setdefault("seen", {})
    hours_back = compute_hours_back()

    run_started_utc = datetime.now(timezone.utc).isoformat()
    new_candidates = fetch_new_jobs(set(seen_map.keys()), hours_back)
    new_matches = evaluate_jobs_with_ai(new_candidates, cv_text, seen_map)

    new_ids = {m["id"] for m in new_matches}
    existing_matches = state.get("matched_jobs", [])
    combined_matches = new_matches + [m for m in existing_matches if m["id"] not in new_ids]

    state["last_run_utc"] = run_started_utc
    state["matched_jobs"] = combined_matches
    save_state(state)

    generate_dashboard(combined_matches, run_started_utc)
    subject, body = build_email(new_matches, hours_back)
    send_email(subject, body)
    print(f"Done. Candidates checked: {len(new_candidates)} | AI matches: {len(new_matches)}")


if __name__ == "__main__":
    main()
