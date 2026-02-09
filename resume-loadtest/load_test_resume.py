#!/usr/bin/env python3
"""
Resume Stress Test (Signed URL, Fully Async)
- For each pipeline:
  1) POST  /loadtest/applicants      -> { applicantId, uploadUrl, uploadHeaders? }
  2) PUT   uploadUrl (signed)        -> PDF bytes  (includes x-goog-meta- headers if provided)
  3) PATCH Firestore: resumeURL      -> gs://<bucket>/jobs/<job>/applicants/<id>/resume.pdf
  4) POST  /invite/extract-photo     -> { jobId, applicantId } (returns download_url)
  5) PATCH Firestore: photoURL       -> <download_url>
  6) GET   /analyze/applicants/{id}  -> server updates Firestore; client records status
"""

import os
import csv
import time
import json
import asyncio
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any

import aiohttp
from google.oauth2 import service_account
from google.auth.transport.requests import Request as GARequest

from contextlib import suppress

# =========================
# Configuration
# =========================
BASE_URL_RESUME = os.getenv("BASE_URL_RESUME", "https://your-resume-service.run.app")
AIP_AUTHENTICATION_KEY = os.getenv("AIP_AUTHENTICATION_KEY", "REPLACE_ME")
JOB_ID = os.getenv("JOB_ID", "YOUR_JOB_ID")
STORAGE_BUCKET = os.getenv("STORAGE_BUCKET", "your-project.firebasestorage.app")  # informational + resumeURL
DATA_ROOT = os.getenv("DATA_ROOT", "../data/resumes")
PROJECT_ID = os.getenv("PROJECT_ID", "your-project-id").strip()  # required for Firestore REST
GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "").strip()

# Concurrency sweep
NUM_REQUESTS = [50, 100, 150, 200, 250, 300]

# Networking knobs
API_TOTAL_TIMEOUT_SEC = int(os.getenv("API_TOTAL_TIMEOUT_SEC", "600"))
API_CONNECT_TIMEOUT_SEC = int(os.getenv("API_CONNECT_TIMEOUT_SEC", "30"))
AIOHTTP_LIMIT = int(os.getenv("AIOHTTP_LIMIT", "1000"))
AIOHTTP_LIMIT_PER_HOST = int(os.getenv("AIOHTTP_LIMIT_PER_HOST", "500"))

# Retry knobs
RETRIES = int(os.getenv("RETRIES", "3"))
BACKOFF_BASE_MS = int(os.getenv("BACKOFF_BASE_MS", "250"))  # exponential backoff base

CSV_PREFIX = "resume_load_test"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

HEALTH_PATH = "/health"

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("resume-signed-stress")

HEADERS_AUTH_JSON = {
    "Authorization": AIP_AUTHENTICATION_KEY,
    "Content-Type": "application/json",
}

# =========================
# OAuth (Firestore REST)
# =========================
_GOOGLE_TOKEN = {"access_token": None, "exp_ts": 0.0}

def _get_access_token_sync() -> str:
    if not GOOGLE_APPLICATION_CREDENTIALS or not Path(GOOGLE_APPLICATION_CREDENTIALS).is_file():
        raise RuntimeError("GOOGLE_APPLICATION_CREDENTIALS not set or file not found")
    creds = service_account.Credentials.from_service_account_file(
        GOOGLE_APPLICATION_CREDENTIALS,
        scopes=["https://www.googleapis.com/auth/datastore"],
    )
    if not creds.valid:
        creds.refresh(GARequest())
    token = creds.token
    # expiry padding: refresh earlier next time
    exp_ts = time.time() + max(3000, 0)  # conservative
    _GOOGLE_TOKEN["access_token"] = token
    _GOOGLE_TOKEN["exp_ts"] = exp_ts
    return token

async def get_access_token() -> str:
    now = time.time()
    if _GOOGLE_TOKEN["access_token"] and _GOOGLE_TOKEN["exp_ts"] - 300 > now:
        return _GOOGLE_TOKEN["access_token"]
    # refresh in thread to avoid blocking loop
    return await asyncio.to_thread(_get_access_token_sync)

async def patch_applicant_fields(session: aiohttp.ClientSession, applicant_id: str, fields: Dict[str, str]) -> int:
    """
    PATCH specific fields on applicants/{applicant_id} using Firestore REST.
    `fields` is a simple dict of { fieldName: stringValue }.
    Returns HTTP status code.
    """
    if not PROJECT_ID:
        logger.error("PROJECT_ID env not set; cannot PATCH Firestore.")
        return 0

    url = f"https://firestore.googleapis.com/v1/projects/{PROJECT_ID}/databases/(default)/documents/applicants/{applicant_id}"
    body = {"fields": {k: {"stringValue": v} for k, v in fields.items()}}
    params = [("currentDocument.exists", "true")]
    for k in fields.keys():
        params.append(("updateMask.fieldPaths", k))

    token = await get_access_token()
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    async with session.patch(url, headers=headers, params=params, json=body) as resp:
        txt = await resp.text()
        if resp.status not in (200, 201):
            logger.warning("PATCH applicants/%s failed: %d %s", applicant_id, resp.status, txt[:300])
        else:
            logger.debug("PATCH applicants/%s OK", applicant_id)
        return resp.status

# =========================
# Utilities
# =========================
def load_pdf_pool(root: str) -> List[Tuple[str, bytes]]:
    """Preload all PDFs into memory: returns list of (filename, bytes)."""
    p = Path(root)
    pdfs = sorted([f for f in p.glob("**/*.pdf") if f.is_file()])
    if not pdfs:
        raise FileNotFoundError(f"No PDF files found under {p.resolve()}")
    pool = [(f.name, f.read_bytes()) for f in pdfs]
    logger.info("Preloaded %d PDFs from %s (total ~%d KB)",
                len(pool), p.resolve(), sum(len(b) for _, b in pool) // 1024)
    return pool

def pick_pdf(pool: List[Tuple[str, bytes]], i: int) -> Tuple[str, bytes]:
    return pool[i % len(pool)]

async def retry_http(coro_fn, *, retries=RETRIES, backoff_base_ms=BACKOFF_BASE_MS, label=""):
    attempt = 0
    last_error = None
    while attempt < retries:
        try:
            return await coro_fn()
        except Exception as e:
            last_error = e
            await asyncio.sleep((backoff_base_ms / 1000.0) * (2 ** attempt))
            attempt += 1
    raise last_error or RuntimeError(f"{label} failed after {retries} retries")

# =========================
# API Calls (all aiohttp)
# =========================
async def api_create_applicant_get_signed_url(session: aiohttp.ClientSession, job_id: str, file_name: str, size_bytes: int) -> Dict[str, Any]:
    url = f"{BASE_URL_RESUME}/loadtest/applicants"
    payload = {"jobId": job_id, "fileName": file_name, "fileSize": size_bytes}
    async def _do():
        async with session.post(url, headers=HEADERS_AUTH_JSON, json=payload) as resp:
            text = await resp.text()
            try:
                data = json.loads(text)
            except Exception:
                logger.error("Non-JSON from /loadtest/applicants: %s", text)
                data = {"raw_text": text}
            return {"status": resp.status, "data": data, "raw": text}
    return await retry_http(_do, label="create_applicant_get_signed_url")

async def put_signed_url(session, signed_url: str, pdf_bytes: bytes, meta_headers: dict) -> Dict[str, Any]:
    headers = {"Content-Type": "application/pdf", **(meta_headers or {})}
    async with session.put(signed_url, data=pdf_bytes, headers=headers) as resp:
        text = await resp.text()
        return {"status": resp.status, "text": text}

async def api_extract_photo(session: aiohttp.ClientSession, job_id: str, applicant_id: str) -> Dict[str, Any]:
    url = f"{BASE_URL_RESUME}/invite/extract-photo"
    payload = {"jobId": job_id, "applicantId": applicant_id}
    async def _do():
        async with session.post(url, headers=HEADERS_AUTH_JSON, json=payload) as resp:
            txt = await resp.text()
            try:
                data = json.loads(txt)
            except Exception:
                data = {"raw_text": txt}
            return {"status": resp.status, "data": data}
    return await retry_http(_do, label="extract_photo")

async def api_analyze(session: aiohttp.ClientSession, applicant_id: str) -> Dict[str, Any]:
    url = f"{BASE_URL_RESUME}/analyze/applicants/{applicant_id}"
    async def _do():
        async with session.get(url, headers=HEADERS_AUTH_JSON) as resp:
            text = await resp.text()
            return {"status": resp.status, "text": text}
    return await retry_http(_do, label="analyze_resume")

# =========================
# One pipeline
# =========================
async def process_single_applicant(i: int, pdf_pool, session: aiohttp.ClientSession) -> dict:
    file_name, pdf_bytes = pick_pdf(pdf_pool, i)
    size_bytes = len(pdf_bytes)

    result: Dict[str, Any] = {
        "request_id": i,
        "file_name": file_name,
        "bytes": size_bytes,
    }

    # 1) create
    t0 = time.perf_counter()
    create_resp = await api_create_applicant_get_signed_url(session, JOB_ID, file_name, size_bytes)
    t1 = time.perf_counter()
    result["create_status"] = create_resp.get("status")
    result["create_ms"] = int((t1 - t0) * 1000)

    data = create_resp.get("data") or {}
    applicant_id = data.get("applicantId")
    upload_url = data.get("uploadUrl")
    upload_headers = data.get("uploadHeaders", {})  # x-goog-meta- headers (if server included)

    if create_resp.get("status") != 200 or not applicant_id or not upload_url:
        result.update({"ok": False, "stage": "create", "error": f"create_status={create_resp.get('status')}, data={data}"})
        return result
    result["applicant_id"] = applicant_id

    # 2) upload
    t0 = time.perf_counter()
    up = await put_signed_url(session, upload_url, pdf_bytes, upload_headers)
    t1 = time.perf_counter()
    result["upload_status"] = up.get("status")
    result["upload_ms"] = int((t1 - t0) * 1000)
    if up.get("status") not in (200, 201):
        result.update({"ok": False, "stage": "upload", "error": f"upload_status={up.get('status')}, body={up.get('text')}"})
        return result

    # 3) PATCH resumeURL BEFORE analyze
    from urllib.parse import quote

    # Build Firebase-style HTTPS URL
    encoded_path = quote(f"jobs/{JOB_ID}/applicants/{applicant_id}/resume.pdf", safe="")
    resume_url = f"https://firebasestorage.googleapis.com/v0/b/{STORAGE_BUCKET}/o/{encoded_path}?alt=media"

    # Patch Firestore applicant doc
    patch_status = await patch_applicant_fields(session, applicant_id, {"resumeURL": resume_url})

    # 4) extract photo
    t0 = time.perf_counter()
    ex = await api_extract_photo(session, JOB_ID, applicant_id)
    t1 = time.perf_counter()
    result["extract_status"] = ex.get("status")
    result["extract_ms"] = int((t1 - t0) * 1000)

    photo_url = None
    if ex.get("status") == 200 and isinstance(ex.get("data"), dict):
        photo_url = ex["data"].get("download_url")
        result["photo_url"] = photo_url
        if photo_url:
            # 5) PATCH photoURL BEFORE analyze
            await patch_applicant_fields(session, applicant_id, {"photoURL": photo_url})

    # 6) analyze
    t0 = time.perf_counter()
    an = await api_analyze(session, applicant_id)
    t1 = time.perf_counter()
    result["analyze_status"] = an.get("status")
    result["analyze_ms"] = int((t1 - t0) * 1000)

    result["ok"] = (result.get("extract_status") == 200 and result.get("analyze_status") == 200)
    result["stage"] = "done" if result["ok"] else "analyze"
    return result

# =========================
# CSV writers
# =========================
def write_csvs(results: List[Dict[str, Any]], concurrency: int, total_seconds: float):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    detailed = f"{CSV_PREFIX}_detailed_{concurrency}_{ts}.csv"
    summary = f"{CSV_PREFIX}_summary_{concurrency}_{ts}.csv"

    # Detailed
    with open(detailed, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "request_id", "applicant_id", "file_name", "bytes",
                "create_status", "create_ms",
                "upload_status", "upload_ms",
                "extract_status", "extract_ms",
                "analyze_status", "analyze_ms",
                "photo_url", "ok", "stage", "error",
            ],
        )
        writer.writeheader()
        for r in results:
            writer.writerow({k: r.get(k) for k in writer.fieldnames})

    # Summary
    ok = [r for r in results if r.get("ok")]
    def series(key): return [r[key] for r in results if r.get(key) is not None]
    def avg(xs): return sum(xs)/len(xs) if xs else 0
    def mn(xs): return min(xs) if xs else 0
    def mx(xs): return max(xs) if xs else 0

    upload = series("upload_ms")
    extract = series("extract_ms")
    analyze = series("analyze_ms")

    with open(summary, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Metric", "Value"])
        w.writerow(["Bucket", STORAGE_BUCKET])
        w.writerow(["Job ID", JOB_ID])
        w.writerow(["Total Requests", len(results)])
        w.writerow(["Run Seconds", f"{total_seconds:.2f}"])
        w.writerow(["Success", len(ok)])
        w.writerow(["Failed", len(results) - len(ok)])
        w.writerow(["Upload Avg (ms)", f"{avg(upload):.2f}"])
        w.writerow(["Upload Min (ms)", mn(upload)])
        w.writerow(["Upload Max (ms)", mx(upload)])
        w.writerow(["Extract Avg (ms)", f"{avg(extract):.2f}"])
        w.writerow(["Extract Min (ms)", mn(extract)])
        w.writerow(["Extract Max (ms)", mx(extract)])
        w.writerow(["Analyze Avg (ms)", f"{avg(analyze):.2f}"])
        w.writerow(["Analyze Min (ms)", mn(analyze)])
        w.writerow(["Analyze Max (ms)", mx(analyze)])

    logger.info("CSV saved: %s, %s", detailed, summary)

# =========================
# Run one round
# =========================
async def run_one_round(concurrency: int, pdf_pool: List[Tuple[str, bytes]]) -> List[Dict[str, Any]]:
    timeout = aiohttp.ClientTimeout(total=API_TOTAL_TIMEOUT_SEC, connect=API_CONNECT_TIMEOUT_SEC)
    connector = aiohttp.TCPConnector(limit=AIOHTTP_LIMIT, limit_per_host=AIOHTTP_LIMIT_PER_HOST, ttl_dns_cache=300, use_dns_cache=True)

    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        tasks = [process_single_applicant(i, pdf_pool, session) for i in range(concurrency)]
        logger.info("🚀 Running round with concurrency = %d", concurrency)
        start = time.time()
        results = await asyncio.gather(*tasks, return_exceptions=False)
        duration = time.time() - start

    ok = sum(1 for r in results if r.get("ok"))
    logger.info("✅ Round %d: %d/%d success in %.2fs", concurrency, ok, concurrency, duration)
    write_csvs(results, concurrency, duration)
    return results



async def _health_ok(session: aiohttp.ClientSession, url: str) -> bool:
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
            if resp.status != 200:
                return False
            with suppress(Exception):
                data = await resp.json(content_type=None)  # tolerate missing/odd content-type
                return isinstance(data, dict) and data.get("status") == "ok"
            # Fallback if not JSON
            text = await resp.text()
            return "ok" in text.lower()
    except Exception:
        return False

async def wait_for_health(timeout_s: int = 90, interval_s: float = 2.0) -> bool:
    """Poll /health until it returns {'status':'ok'} or timeout."""
    base_url = BASE_URL_RESUME
    health_url = f"{base_url.rstrip('/')}{HEALTH_PATH}"
    start = time.monotonic()
    attempt = 0
    async with aiohttp.ClientSession() as session:
        while True:
            attempt += 1
            ok = await _health_ok(session, health_url)
            if ok:
                logger.info("✅ Resume API is healthy at %s", health_url)
                return True
            elapsed = time.monotonic() - start
            if elapsed >= timeout_s:
                logger.error("❌ Health check timed out after %.1fs (last attempt #%d) at %s",
                             elapsed, attempt, health_url)
                return False
            sleep_for = interval_s
            logger.info("⏳ Waiting for service health (attempt #%d). Retrying in %.1fs...",
                        attempt, sleep_for)
            await asyncio.sleep(sleep_for)




# =========================
# Main
# =========================
async def main():
    logger.info("🔥 Starting Resume Stress Test (Signed URL, Fully Async)")
    if not PROJECT_ID:
        logger.warning("PROJECT_ID is not set; Firestore PATCH will fail.")
    if not GOOGLE_APPLICATION_CREDENTIALS:
        logger.warning("GOOGLE_APPLICATION_CREDENTIALS is not set; Firestore PATCH will fail.")

    # ✅ Block here until /health reports {"status":"ok"} (or abort on timeout)
    # This ensures the service is ready before we start load testing.
    # To avoid "cold start" issue
    ready = await wait_for_health(timeout_s=200, interval_s=2.5)
    if not ready:
        logger.error("Aborting: Resume API never became healthy.")
        return  # or: raise SystemExit(1)
    else:
        logger.info("✅ Resume API is healthy; proceeding with load test.")

    pdf_pool = load_pdf_pool(DATA_ROOT)

    for concurrency in NUM_REQUESTS:
        await run_one_round(concurrency, pdf_pool)
        if concurrency != NUM_REQUESTS[-1]:
            logger.info("⏳ Cooling down 5s before next round...")
            await asyncio.sleep(5)

if __name__ == "__main__":
    asyncio.run(main())
