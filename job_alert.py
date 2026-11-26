"""
Daily AI-powered job alert:
1. Stage A: Cheaply fetch LinkedIn job IDs/URLs via Apify (no descriptions) since last run.
2. Filter out already-seen job IDs stored in `seen_jobs.json`.
3. Stage B: Fetch full descriptions ONLY for brand-new job URLs.
4. Score each new job against `CV_TEXT` secret using Gemini 2.5 Flash + generate tailored bullets.
5. Update `seen_jobs.json`, generate an interactive GitHub Pages dashboard (`docs/index.html`),
   and email the new matches.
"""

import json
import math
import os
import smtplib
import ssl
from datetime import datetime, timezone
from email.message import EmailMessage
from html import escape
from pathlib import Path
from urllib.parse import quote_plus

from apify_client import ApifyClient
from google import genai
from pydantic import BaseModel, Field

# ---- Settings -------------------------------------------------------------
ACTOR_ID = "mighty_monk/linkedin-jobs-scraper"

DEFAULT_QUERIES = [
    '"AI" AND ("Program Manager" OR "Project Manager" OR "Transformation" OR "Adoption")',
    '"IT Service" AND ("Manager" OR "Lead" OR "Owner") NOT "Junior" NOT "Intern"',
    '"Asset" AND ("Manager" OR "Owner" OR "Lead") AND ("IT" OR "Discovery" OR "Configuration")',
    '"Service Delivery Manager" NOT "Junior" NOT "Sales"',
]


def _env(name: str, default: str) -> str:
    value = os.environ.get(name, "").strip()
    return value or default


def _env_list(name: str, default: list[str]) -> list[str]:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    parts = raw.replace("\n", ",").split(",")
    return [p.strip() for p in parts if p.strip()]


QUERIES = _env_list("SEARCH_QUERIES", DEFAULT_QUERIES)
LOCATION = _env("SEARCH_LOCATION", "Malaysia")
MAX_JOBS = int(_env("MAX_JOBS", "30"))
DEFAULT_HOURS_BACK = int(_env("HOURS_BACK", "24"))
MIN_MATCH_SCORE = int(_env("MIN_MATCH_SCORE", "7"))

STATE_FILE = Path("seen_jobs.json")
DASHBOARD_FILE = Path("docs/index.html")
# ---------------------------------------------------------------------------


class JobEvaluation(BaseModel):
    score: int = Field(description="Match score from 1 to 10 against candidate CV")
    is_match: bool = Field(description="True if score >= MIN_MATCH_SCORE and seniority/domain fit")
    reason: str = Field(description="1-2 concise sentences on why this fits or key gaps")
    tailored_bullets: list[str] = Field(
        description="3 tailored resume bullet points aligned to this job description"
    )


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


def compute_hours_back(last_run_utc: str | None) -> int:
    if not last_run_utc:
        return DEFAULT_HOURS_BACK
    try:
        last_dt = datetime.fromisoformat(last_run_utc)
        elapsed_hours = (datetime.now(timezone.utc) - last_dt).total_seconds() / 3600.0
        return max(2, min(72, math.ceil(elapsed_hours + 1)))
    except Exception:
        return DEFAULT_HOURS_BACK


def search_url(query: str, hours_back: int) -> str:
    return (
        "https://www.linkedin.com/jobs/search/"
        f"?keywords={quote_plus(query)}&location={quote_plus(LOCATION)}"
        f"&f_TPR=r{hours_back * 3600}&sortBy=DD"
    )


def _run_apify(client: ApifyClient, urls: list[str], include_desc: bool, max_jobs: int) -> list[dict]:
    run_input = {
        "searchUrls": urls,
        "maxJobs": max_jobs,
        "maxConcurrency": 2,
        "requestDelayMs": 1500,
        "includeDescription": include_desc,
        "proxyConfiguration": {"useApifyProxy": True},
    }
    run = client.actor(ACTOR_ID).call(run_input=run_input)
    dataset_id = run["defaultDatasetId"] if isinstance(run, dict) else run.default_dataset_id
    return client.dataset(dataset_id).list_items().items


def fetch_new_jobs(seen_ids: set[str], hours_back: int) -> list[dict]:
    client = ApifyClient(os.environ["APIFY_TOKEN"])

    # Stage A: Cheap metadata-only scan (no descriptions scraped)
    urls = [search_url(q, hours_back) for q in QUERIES]
    print(f"Stage A: Scanning {len(urls)} search URL(s) for last {hours_back}h (metadata only)...")
    shallow_items = _run_apify(client, urls, include_desc=False, max_jobs=MAX_JOBS)

    new_candidates = []
    batch_seen = set()
    for item in shallow_items:
        job_id = str(item.get("jobId") or item.get("id") or item.get("jobUrl") or "").strip()
        job_url = (item.get("jobUrl") or item.get("applyUrl") or "").strip()
        if not job_id or not job_url or job_id in seen_ids or job_id in batch_seen:
            continue
        if not item.get("title"):
            continue
        batch_seen.add(job_id)
        item["_clean_id"] = job_id
        item["_clean_url"] = job_url
        new_candidates.append(item)

    print(f"Stage A found {len(shallow_items)} total -> {len(new_candidates)} unseen job(s).")
    if not new_candidates:
        return []

    # Stage B: Scrape full descriptions ONLY for brand-new job URLs
    new_urls = [c["_clean_url"] for c in new_candidates]
    print(f"Stage B: Fetching full descriptions for {len(new_urls)} unseen job(s)...")
    detailed_items = _run_apify(client, new_urls, include_desc=True, max_jobs=len(new_urls))

    detail_by_id = {}
    for d in detailed_items:
        did = str(d.get("jobId") or d.get("id") or d.get("jobUrl") or "").strip()
        if did:
            detail_by_id[did] = d

    enriched = []
    for c in new_candidates:
        full = detail_by_id.get(c["_clean_id"]) or detail_by_id.get(c["_clean_url"]) or c
        full["_clean_id"] = c["_clean_id"]
        full["_clean_url"] = c["_clean_url"]
        enriched.append(full)

    return enriched


def evaluate_jobs_with_ai(jobs: list[dict], cv_text: str, seen_map: dict) -> list[dict]:
    if not jobs:
        return []

    ai_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    matched = []

    for job in jobs:
        job_id = job["_clean_id"]
        title = job.get("title", "Unknown Role")
        company = job.get("company") or job.get("companyName") or "Unknown Company"
        desc = job.get("description") or job.get("descriptionText") or ""

        if not desc.strip():
            seen_map[job_id] = {"status": "no_description", "date": today_str, "title": title}
            continue

        prompt = f"""
Evaluate how well this candidate matches the job posting below.
Be realistic and selective:
- Score >= {MIN_MATCH_SCORE} ONLY if the candidate's seniority, leadership scope, and core technical/domain experience genuinely align.
- Penalize junior roles, pure sales/pre-sales quotas, or unrelated domain stacks.
- Write 3 tailored, metric-oriented resume bullet points that position the candidate's real background for this specific role.

CANDIDATE CV:
{cv_text[:8000]}

JOB TITLE: {title}
COMPANY: {company}
LOCATION: {job.get("location", LOCATION)}
JOB DESCRIPTION:
{desc[:7000]}
"""
        try:
            response = ai_client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config={
                    "response_mime_type": "application/json",
                    "response_schema": JobEvaluation,
                    "temperature": 0.2,
                },
            )
            result = json.loads(response.text)
            score = int(result.get("score", 0))
            is_match = bool(result.get("is_match")) and score >= MIN_MATCH_SCORE

            if is_match:
                seen_map[job_id] = {"status": f"matched_{score}", "date": today_str, "title": title}
                matched.append({
                    "id": job_id,
                    "title": title,
                    "company": company,
                    "location": job.get("location") or LOCATION,
                    "posted_at": str(job.get("postedAt") or today_str),
                    "found_date": today_str,
                    "url": job["_clean_url"],
                    "score": score,
                    "reason": result.get("reason", ""),
                    "bullets": result.get("tailored_bullets", []),
                })
            else:
                seen_map[job_id] = {"status": f"rejected_{score}", "date": today_str, "title": title}
        except Exception as exc:
            print(f"AI evaluation failed for '{title}': {exc}")

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
            f"<p>No new jobs scored &ge; {MIN_MATCH_SCORE}/10 in the last {hours_back} hours.</p>"
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
        f"<p>Found <b>{len(new_matches)}</b> role(s) matching your CV (&ge; {MIN_MATCH_SCORE}/10):</p>"
        f"{pages_link}"
        f'{"".join(cards)}'
    )
    return subject, body


def send_email(subject: str, html_body: str) -> None:
    sender = os.environ["GMAIL_ADDRESS"]
    recipient = os.environ.get("EMAIL_TO", "").strip() or sender

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = recipient
    msg.set_content("Your email client does not support HTML.")
    msg.add_alternative(html_body, subtype="html")

    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=ssl.create_default_context()) as smtp:
        smtp.login(sender, os.environ["GMAIL_APP_PASSWORD"])
        smtp.send_message(msg)


def generate_dashboard(all_matches: list[dict], last_run_utc: str) -> None:
    DASHBOARD_FILE.parent.mkdir(parents=True, exist_ok=True)
    repo = os.environ.get("GITHUB_REPOSITORY", "YOUR_USERNAME/YOUR_REPO")
    owner, repo_name = repo.split("/", 1) if "/" in repo else ("YOUR_USERNAME", "YOUR_REPO")
    jobs_json = json.dumps(all_matches, ensure_ascii=False)

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
    header {{ display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center; gap: 12px; margin-bottom: 20px; }}
    h1 {{ margin: 0; font-size: 24px; }}
    .sub {{ color: var(--muted); font-size: 13px; margin-top: 4px; }}
    .controls {{ display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 18px; }}
    input, select, button {{ padding: 9px 12px; border: 1px solid var(--border); border-radius: 8px; font-size: 14px; background: #fff; }}
    input {{ flex: 1; min-width: 220px; }}
    button.primary {{ background: var(--accent); color: #fff; border: none; font-weight: 600; cursor: pointer; }}
    button.primary:disabled {{ opacity: 0.6; cursor: not-allowed; }}
    button.secondary {{ background: #fff; color: var(--text); cursor: pointer; }}
    #statusBanner {{ display: none; padding: 12px 14px; border-radius: 8px; background: #eff6ff; border: 1px solid #bfdbfe; color: #1e40af; font-size: 14px; margin-bottom: 16px; }}
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
      <div style="display:flex; gap:8px;">
        <button class="secondary" onclick="resetToken()" title="Update GitHub Token">API Key</button>
        <button id="runBtn" class="primary" onclick="triggerScan()">&#9654; Run Scan Now</button>
      </div>
    </header>

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

      const btn = document.getElementById("runBtn");
      btn.disabled = true;
      setBanner("Triggering GitHub Actions workflow...");

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
            body: JSON.stringify({{ ref: "main" }})
          }}
        );

        if (!dispatchRes.ok) {{
          throw new Error(`GitHub API error (${{dispatchRes.status}}). Check repo name and PAT permissions.`);
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
    hours_back = compute_hours_back(state.get("last_run_utc"))

    run_started_utc = datetime.now(timezone.utc).isoformat()
    new_candidates = fetch_new_jobs(set(seen_map.keys()), hours_back)
    new_matches = evaluate_jobs_with_ai(new_candidates, cv_text, seen_map)

    existing_matches = state.get("matched_jobs", [])
    existing_ids = {m["id"] for m in existing_matches}
    combined_matches = new_matches + [m for m in existing_matches if m["id"] not in existing_ids]

    state["last_run_utc"] = run_started_utc
    state["matched_jobs"] = combined_matches
    save_state(state)

    generate_dashboard(combined_matches, run_started_utc)
    subject, body = build_email(new_matches, hours_back)
    send_email(subject, body)
    print(f"Done. New candidates checked: {len(new_candidates)} | New AI matches: {len(new_matches)}")


if __name__ == "__main__":
    main()
