"""
Daily job alert: scrape LinkedIn jobs posted in the last 24 hours via Apify,
then email the results to yourself.

Setup:
    pip install apify-client
    export APIFY_TOKEN="..."            # Apify Console > Settings > API & Integrations
    export GMAIL_ADDRESS="you@gmail.com"
    export GMAIL_APP_PASSWORD="..."     # Google Account > Security > App passwords
    python job_alert.py

Optional: EMAIL_TO (defaults to GMAIL_ADDRESS)
"""

import os
import smtplib
import ssl
from datetime import date
from email.message import EmailMessage
from html import escape
from urllib.parse import quote_plus

from apify_client import ApifyClient

# ---- Settings -------------------------------------------------------------
# Each can be set from GitHub (Settings > Variables, or the "Run workflow" form).
# The values below are only defaults used when nothing is set.
ACTOR_ID = "mighty_monk/linkedin-jobs-scraper"
DEFAULT_QUERIES = [
    "AI program manager",
    "AI project manager",
    "AI adoption lead",
    "AI transformation lead",
    "AI transformation manager",
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


QUERIES = _env_list("SEARCH_QUERIES", DEFAULT_QUERIES)  # comma or one per line
LOCATION = _env("SEARCH_LOCATION", "Malaysia")
MAX_JOBS = int(_env("MAX_JOBS", "100"))                  # cap per run (controls cost)
HOURS_BACK = int(_env("HOURS_BACK", "24"))               # how recent the jobs must be
INCLUDE_DESCRIPTION = False                              # False = cheaper and faster
# -----------------------------------------------------------------------------


def search_url(query: str) -> str:
    # f_TPR=r86400 -> posted in the past 24 hours; sortBy=DD -> most recent first
    return (
        "https://www.linkedin.com/jobs/search/"
        f"?keywords={quote_plus(query)}&location={quote_plus(LOCATION)}"
        f"&f_TPR=r{HOURS_BACK * 3600}&sortBy=DD"
    )


def fetch_jobs() -> list[dict]:
    client = ApifyClient(os.environ["APIFY_TOKEN"])
    run_input = {
        "searchUrls": [search_url(q) for q in QUERIES],
        "maxJobs": MAX_JOBS,
        "maxConcurrency": 2,
        "requestDelayMs": 1500,
        "includeDescription": INCLUDE_DESCRIPTION,
        "proxyConfiguration": {"useApifyProxy": True},
    }
    run = client.actor(ACTOR_ID).call(run_input=run_input)
    items = client.dataset(run["defaultDatasetId"]).list_items().items

    # De-duplicate (the same job can match several queries)
    seen, jobs = set(), []
    for item in items:
        key = item.get("jobId") or item.get("jobUrl")
        if not key or key in seen or not item.get("title"):
            continue
        seen.add(key)
        jobs.append(item)
    return jobs


def build_email(jobs: list[dict]) -> tuple[str, str]:
    today = date.today().strftime("%d %b %Y")
    subject = f"Job alert {today}: {len(jobs)} new listing(s)"

    if not jobs:
        return subject, f"<p>No new jobs in the last {HOURS_BACK} hours for your searches.</p>"

    rows = []
    for j in jobs:
        title = escape(j.get("title", ""))
        company = escape(j.get("company") or j.get("companyName") or "")
        location = escape(j.get("location", ""))
        posted = escape(str(j.get("postedAt", "")))
        url = escape(j.get("jobUrl") or j.get("applyUrl") or "#")
        rows.append(
            f'<li style="margin-bottom:12px">'
            f'<a href="{url}"><b>{title}</b></a><br>'
            f"{company} &middot; {location}"
            f'{" &middot; " + posted if posted else ""}</li>'
        )
    body = (
        f"<p>{len(jobs)} job(s) posted in the last {HOURS_BACK} hours:</p>"
        f'<ul style="padding-left:18px">{"".join(rows)}</ul>'
    )
    return subject, body


def send_email(subject: str, html_body: str) -> None:
    sender = os.environ["GMAIL_ADDRESS"]
    recipient = os.environ.get("EMAIL_TO", sender)

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = recipient
    msg.set_content("Your email client does not support HTML.")
    msg.add_alternative(html_body, subtype="html")

    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=ssl.create_default_context()) as smtp:
        smtp.login(sender, os.environ["GMAIL_APP_PASSWORD"])
        smtp.send_message(msg)


def main() -> None:
    jobs = fetch_jobs()
    subject, body = build_email(jobs)
    send_email(subject, body)
    print(f"Sent {len(jobs)} job(s).")


if __name__ == "__main__":
    main()
