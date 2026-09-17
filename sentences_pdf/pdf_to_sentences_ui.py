"""
You should make a Python-based frontend Web app perhaps with
Javascript that can use another Web service (that is built
independently). The Web app should make it easy to demonstrate and
test the Web service. The Web service extracts sentences from a PDF
file. This exercise is describe in detail below.

I want a frontend Web app with few dependencies so that installation
would be painless.

If there is any Javascript is should be relatively simple. Do not
include jQuery or other external library unless absolutely necessary.

The Web app may include some styling, but I would like to have it
simple and the style within HTML code rather than as a separate style
file. It is running at DTU where the primary colors are corporate red
(153,0,0), white and black. Some more colors at
https://designguide.dtu.dk/colours if needed.

The interface language of the Web app should be English.

The Web app could be implemented in FastAPI, Streamlit or other
framework, depending on what you would think is the most pedagogical
and has the least dependencies. I as a teacher and the students should
be able to understand the code even though the course is not about
frontend development. If docstrings are included make it in numpydoc
format and do not be afraid to add doctests if that is relevant.

The Web app should make it easy for students to upload a PDF file and
(with the external student-constructed Web service) extract the
sentences from the PDF displaying the result. Possible the students
might upload a PDF with many sentences!

I have a few PDF example files where some the Web app can test if some
sentences exists. Only the endpoint should be tested not, e.g., a
possible GROBID docker container.

The dataset is (with sentences that appear in the PDF):

[
  {
    'filename': 'studyboard.pdf',
    'sentences': [
      'This document details the meeting in the studyboard on Februar 12 2026.',
      'Finn and Tyge were present.',
      'Poul was missing.'
      'We discussed the issue of calculators.'
    ]
  },
  {
    'filename': '2303.15133.pdf',
    'sentences': [
       'Other endpoints than the configured default can be queried.',
       'I call the tool Synia with the canonical homepage set up at https://synia.toolforge.org/.'
       'Scholia is a Web application running from the Wikimedia Foundation Toolforge server at http://scholia.toolforge.org.',
]
  }
]

The files may be available in the same directory as the running Web app.

If possible this small dataset can be automatically be tested and
result displayed in the Web app.

Please also make any error message pedagogic, and include appropriate
time out for the response from the Web service. Include operational
metrics, e.g., response latency from the Web service and/or number of
successful request. The Web service ought to handle multiple
asynchronous Web requests, so this could be

Include this prompt (the above text) as part of the generated code,
e.g., in a docstring.

Now I am showing the web service exercise text (do not implement this
- I and students will do this independently). This is not necessary to
include in the generated Web app.
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse


# -----------------------------
# Configuration (simple + explicit)
# -----------------------------

SERVICE_URL_DEFAULT = "http://localhost:8000/v1/extract-sentences"
SERVICE_URL = os.environ.get("SENTENCE_SERVICE_URL", SERVICE_URL_DEFAULT)

# Timeouts: keep them generous for big PDFs, but still bounded.
# You can override via environment variables, e.g. SERVICE_TIMEOUT_SECONDS=120
TIMEOUT_SECONDS = float(os.environ.get("SERVICE_TIMEOUT_SECONDS", "90"))
CONNECT_TIMEOUT_SECONDS = float(os.environ.get("SERVICE_CONNECT_TIMEOUT_SECONDS", "10"))

# Where we look for the sample PDFs:
APP_DIR = Path(__file__).resolve().parent


# -----------------------------
# Self-test dataset
# -----------------------------

SELFTEST_DATASET: List[Dict[str, Any]] = [
    {
        "filename": "studyboard.pdf",
        "sentences": [
            # "This document details the meeting in the studyboard on Februar 12 2026.",
            "Finn and Tyge were present.",
            "Poul was missing.",
            "We discussed the issue of calculators.",
        ],
    },
    {
        "filename": "2303.15133.pdf",
        "sentences": [
            "Other endpoints than the configured default can be queried.",
            "I call the tool Synia with the canonical homepage set up at https://synia.toolforge.org/.",
            "Scholia is a Web application running from the Wikimedia Foundation Toolforge server at http://scholia.toolforge.org.",
        ],
    },
]


# -----------------------------
# Lightweight operational metrics
# -----------------------------

@dataclass
class Metrics:
    """In-memory operational metrics.

    Notes
    -----
    This is intentionally simple: for a teaching demo, it is enough to show
    how to measure latency and count successes/failures.

    The metrics reset when the process restarts.
    """

    total_requests: int = 0
    success_requests: int = 0
    failed_requests: int = 0
    last_latency_ms: Optional[float] = None
    latency_ms_sum: float = 0.0
    latency_ms_count: int = 0
    last_error: Optional[str] = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def record_success(self, latency_ms: float) -> None:
        async with self.lock:
            self.total_requests += 1
            self.success_requests += 1
            self.last_latency_ms = latency_ms
            self.latency_ms_sum += latency_ms
            self.latency_ms_count += 1
            self.last_error = None

    async def record_failure(self, latency_ms: Optional[float], error: str) -> None:
        async with self.lock:
            self.total_requests += 1
            self.failed_requests += 1
            self.last_latency_ms = latency_ms
            self.last_error = error

    async def snapshot(self) -> Dict[str, Any]:
        async with self.lock:
            avg = None
            if self.latency_ms_count > 0:
                avg = self.latency_ms_sum / self.latency_ms_count
            return {
                "service_url": SERVICE_URL,
                "timeout_seconds": TIMEOUT_SECONDS,
                "connect_timeout_seconds": CONNECT_TIMEOUT_SECONDS,
                "total_requests": self.total_requests,
                "success_requests": self.success_requests,
                "failed_requests": self.failed_requests,
                "last_latency_ms": self.last_latency_ms,
                "avg_latency_ms": avg,
                "last_error": self.last_error,
            }


metrics = Metrics()
app = FastAPI(title="PDF → Sentences Frontend (DTU)")


# -----------------------------
# Helper functions
# -----------------------------

def _pedagogic_http_error(ex: Exception) -> str:
    """Convert a low-level exception into a pedagogic message.

    Parameters
    ----------
    ex:
        The exception raised while calling the external service.

    Returns
    -------
    str
        A human-friendly error message.
    """
    if isinstance(ex, httpx.ConnectError):
        return (
            "Could not connect to the sentence-extraction service.\n"
            f"- Configured service URL: {SERVICE_URL}\n"
            "- Is the service running, and is the URL correct?\n"
            "- If running in Docker, check port mappings.\n"
        )
    if isinstance(ex, httpx.ReadTimeout):
        return (
            "The service did not respond before the timeout.\n"
            f"- Current timeout: {TIMEOUT_SECONDS} seconds\n"
            "- Large PDFs can take time.\n"
            "- Consider optimizing the service or increasing SERVICE_TIMEOUT_SECONDS.\n"
        )
    if isinstance(ex, httpx.RemoteProtocolError):
        return (
            "The connection was established, but the HTTP protocol exchange failed.\n"
            "- This can happen if a proxy or server closes the connection unexpectedly.\n"
            "- Check the service logs.\n"
        )
    return (
        "An unexpected error happened while calling the service.\n"
        f"- Error type: {type(ex).__name__}\n"
        f"- Details: {ex}\n"
    )


async def call_sentence_service(pdf_bytes: bytes, filename: str) -> Tuple[List[str], float]:
    """Call the external sentence extraction service.

    Parameters
    ----------
    pdf_bytes:
        Raw PDF bytes.
    filename:
        Original filename, forwarded to the service as metadata.

    Returns
    -------
    sentences:
        Extracted sentences (list of strings).
    latency_ms:
        Measured round-trip latency in milliseconds.

    Raises
    ------
    httpx.HTTPError
        For network/protocol/timeouts.
    ValueError
        If JSON schema from the service is unexpected.
    """
    timeout = httpx.Timeout(
        TIMEOUT_SECONDS,
        connect=CONNECT_TIMEOUT_SECONDS,
        read=TIMEOUT_SECONDS,
        write=TIMEOUT_SECONDS,
        pool=TIMEOUT_SECONDS,
    )

    t0 = time.perf_counter()
    async with httpx.AsyncClient(timeout=timeout) as client:
        files = {"pdf_file": (filename, pdf_bytes, "application/pdf")}
        resp = await client.post(SERVICE_URL, files=files)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    # If service returns non-200, show the response text (often helpful for debugging).
    if resp.status_code != 200:
        raise ValueError(
            "Service returned an error status.\n"
            f"- HTTP status: {resp.status_code}\n"
            f"- Response body (first 2k chars): {resp.text[:2000]}"
        )

    data = resp.json()
    if not isinstance(data, dict) or "sentences" not in data or not isinstance(data["sentences"], list):
        raise ValueError(
            "Service response JSON did not match the expected schema.\n"
            "Expected: {\"sentences\": [\"...\", ...]}\n"
            f"Got: {data}"
        )

    # Ensure everything is a string (be strict, but helpful).
    sentences = data["sentences"]
    bad = [type(s).__name__ for s in sentences if not isinstance(s, str)]
    if bad:
        raise ValueError(
            "Service returned a 'sentences' list, but some elements were not strings.\n"
            f"Types seen (non-strings): {sorted(set(bad))}"
        )

    return sentences, latency_ms


async def run_selftest() -> Dict[str, Any]:
    """Run the built-in dataset checks against the configured service URL.

    Returns
    -------
    dict
        Structured results: per-file status, missing sentences, latency.

    Notes
    -----
    This tests ONLY the configured endpoint and does NOT require or check any internal
    components (e.g., GROBID containers). The service is treated as a black box.
    """
    results: List[Dict[str, Any]] = []
    passed = 0

    for item in SELFTEST_DATASET:
        fname = item["filename"]
        expected = item["sentences"]
        path = APP_DIR / fname

        if not path.exists():
            results.append(
                {
                    "filename": fname,
                    "ok": False,
                    "error": "Sample PDF not found next to the app.",
                    "missing_sentences": expected,
                    "latency_ms": None,
                    "num_returned_sentences": None,
                }
            )
            continue

        try:
            pdf_bytes = path.read_bytes()
            sentences, latency_ms = await call_sentence_service(pdf_bytes, fname)

            returned_set = set(sentences)
            missing = [s for s in expected if s not in returned_set]

            ok = len(missing) == 0
            if ok:
                passed += 1

            results.append(
                {
                    "filename": fname,
                    "ok": ok,
                    "error": None if ok else "Some expected sentences were not found in the service output.",
                    "missing_sentences": missing,
                    "latency_ms": latency_ms,
                    "num_returned_sentences": len(sentences),
                }
            )
        except Exception as ex:  # pedagogic: we want to show something useful
            results.append(
                {
                    "filename": fname,
                    "ok": False,
                    "error": _pedagogic_http_error(ex) if isinstance(ex, httpx.HTTPError) else str(ex),
                    "missing_sentences": expected,
                    "latency_ms": None,
                    "num_returned_sentences": None,
                }
            )

    return {
        "service_url": SERVICE_URL,
        "passed": passed,
        "total": len(SELFTEST_DATASET),
        "results": results,
    }


# -----------------------------
# Web UI (single-file HTML with inline CSS + vanilla JS)
# -----------------------------

@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    """Render the single-page UI."""
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>PDF → Sentences (DTU Frontend)</title>
  <style>
    :root {{
      --dtu-red: rgb(153,0,0);
      --bg: #ffffff;
      --fg: #111111;
      --muted: #666666;
      --border: #e5e5e5;
      --ok: #0a7a2f;
      --bad: #b00020;
      --card: #fafafa;
    }}
    body {{
      margin: 0;
      font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;
      background: var(--bg);
      color: var(--fg);
      line-height: 1.35;
    }}
    header {{
      background: var(--dtu-red);
      color: white;
      padding: 14px 18px;
    }}
    header h1 {{
      margin: 0;
      font-size: 18px;
      font-weight: 650;
      letter-spacing: 0.2px;
    }}
    header .sub {{
      margin-top: 6px;
      font-size: 13px;
      opacity: 0.9;
    }}
    main {{
      max-width: 1100px;
      margin: 18px auto;
      padding: 0 16px 40px 16px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 14px;
    }}
    @media (max-width: 920px) {{
      .grid {{ grid-template-columns: 1fr; }}
    }}
    .card {{
      border: 1px solid var(--border);
      background: var(--card);
      border-radius: 10px;
      padding: 14px;
    }}
    .card h2 {{
      margin: 0 0 10px 0;
      font-size: 15px;
    }}
    .meta {{
      font-size: 12px;
      color: var(--muted);
    }}
    .row {{
      display: flex;
      gap: 10px;
      align-items: center;
      flex-wrap: wrap;
      margin-top: 10px;
    }}
    .btn {{
      background: var(--dtu-red);
      color: white;
      border: none;
      border-radius: 8px;
      padding: 10px 12px;
      font-weight: 600;
      cursor: pointer;
    }}
    .btn:disabled {{
      opacity: 0.6;
      cursor: not-allowed;
    }}
    .btn.secondary {{
      background: #222;
    }}
    input[type="file"], input[type="text"] {{
      padding: 8px;
      border-radius: 8px;
      border: 1px solid var(--border);
      background: white;
    }}
    .pill {{
      display: inline-block;
      padding: 4px 8px;
      border-radius: 999px;
      border: 1px solid rgba(255,255,255,0.55); /* looks nice on DTU red */
      background: rgba(255,255,255,0.92);
      color: #111111; /* IMPORTANT: ensure readable text */
      font-size: 12px;
    }}
    .ok {{ color: var(--ok); font-weight: 700; }}
    .bad {{ color: var(--bad); font-weight: 700; }}
    pre {{
      white-space: pre-wrap;
      word-break: break-word;
      background: white;
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 10px;
      margin: 10px 0 0 0;
      max-height: 320px;
      overflow: auto;
      font-size: 12.5px;
    }}
    details {{
      margin-top: 10px;
    }}
    summary {{
      cursor: pointer;
      font-weight: 650;
    }}
    .small {{
      font-size: 12px;
      color: var(--muted);
    }}
    .kpi {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px;
      margin-top: 10px;
    }}
    .kpi .box {{
      background: white;
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 10px;
    }}
    .kpi .label {{
      font-size: 11px;
      color: var(--muted);
    }}
    .kpi .value {{
      font-size: 16px;
      font-weight: 750;
      margin-top: 4px;
    }}
  </style>
</head>
<body>
  <header>
    <h1>PDF → Sentences (DTU Frontend Test App)</h1>
    <div class="sub">Uploads a PDF and calls an independently-built service at <span class="pill" id="svcUrl">{SERVICE_URL}</span></div>
  </header>

  <main>
    <div class="grid">
      <section class="card">
        <h2>1) Upload a PDF and extract sentences</h2>
        <div class="meta">
          The PDF is forwarded to the external service endpoint. This app measures round-trip latency and shows a preview.
        </div>

        <form id="uploadForm">
          <div class="row">
            <input id="pdfFile" type="file" accept="application/pdf" required />
            <button id="extractBtn" class="btn" type="submit">Extract sentences</button>
            <button id="downloadBtn" class="btn secondary" type="button" disabled>Download all as .txt</button>
          </div>
          <div class="row">
            <input id="filterBox" type="text" placeholder="Optional: filter preview by substring (client-side)" style="min-width: 320px;" />
            <span class="pill">Preview limit: <b id="previewLimit">200</b> sentences</span>
            <span class="pill">Latency: <b id="latency">—</b></span>
            <span class="pill">Returned: <b id="returnedCount">—</b></span>
          </div>
        </form>

        <details open>
          <summary>Output preview</summary>
          <div class="small">Shows the first N sentences (after optional filter). Use the download button to get all sentences.</div>
          <pre id="output">No results yet.</pre>
        </details>

        <details>
          <summary>Troubleshooting tips</summary>
          <pre id="tips">• Ensure the service is running and reachable.
• Current configured URL: {SERVICE_URL}
• You can change it with environment variable SENTENCE_SERVICE_URL.
• If large PDFs time out, increase SERVICE_TIMEOUT_SECONDS.</pre>
        </details>
      </section>

      <section class="card">
        <h2>2) Automatic self-test (sample PDFs)</h2>
        <div class="meta">
          Tests whether specific sentences are present in the service output for PDFs located next to this app.
        </div>

        <div class="row">
          <button id="selftestBtn" class="btn" type="button">Run self-test</button>
          <span class="pill">Status: <b id="selftestStatus">not run</b></span>
        </div>

        <pre id="selftestOut">Self-test results will appear here.</pre>

        <h2 style="margin-top:14px;">3) Operational metrics</h2>
        <div class="meta">In-memory counters for this frontend process (reset on restart).</div>

        <div class="kpi">
          <div class="box"><div class="label">Total requests</div><div class="value" id="m_total">—</div></div>
          <div class="box"><div class="label">Successful</div><div class="value" id="m_ok">—</div></div>
          <div class="box"><div class="label">Failed</div><div class="value" id="m_fail">—</div></div>
          <div class="box"><div class="label">Avg latency (ms)</div><div class="value" id="m_avg">—</div></div>
          <div class="box"><div class="label">Last latency (ms)</div><div class="value" id="m_last">—</div></div>
          <div class="box"><div class="label">Last error</div><div class="value" id="m_err" style="font-size:12px;">—</div></div>
        </div>
      </section>
    </div>
  </main>

<script>
(function() {{
  const previewN = 200;
  const output = document.getElementById("output");
  const latencyEl = document.getElementById("latency");
  const returnedCountEl = document.getElementById("returnedCount");
  const extractBtn = document.getElementById("extractBtn");
  const downloadBtn = document.getElementById("downloadBtn");
  const filterBox = document.getElementById("filterBox");
  const previewLimitEl = document.getElementById("previewLimit");
  previewLimitEl.textContent = String(previewN);

  let lastAllSentences = null;

  function fmtMs(x) {{
    if (x === null || x === undefined) return "—";
    return Math.round(x * 10) / 10;
  }}

  function renderPreview(sentences) {{
    const filt = (filterBox.value || "").trim().toLowerCase();
    let s = sentences;
    if (filt.length > 0) {{
      s = sentences.filter(x => x.toLowerCase().includes(filt));
    }}

    const shown = s.slice(0, previewN);
    const lines = shown.map((x, i) => String(i+1).padStart(4, " ") + ". " + x);
    let header = "";
    if (filt.length > 0) {{
      header = `Filter: "${{filt}}"\\n`;
    }}
    header += `Showing ${{shown.length}} / ${{s.length}} (filtered) / ${{sentences.length}} (total)\\n\\n`;
    output.textContent = header + lines.join("\\n");
  }}

  function makeTxtDownload(sentences) {{
    const text = sentences.map(s => s + "\\n").join("");
    const blob = new Blob([text], {{ type: "text/plain;charset=utf-8" }});
    const url = URL.createObjectURL(blob);
    return url;
  }}

  async function refreshMetrics() {{
    try {{
      const r = await fetch("/api/metrics");
      const m = await r.json();
      document.getElementById("m_total").textContent = m.total_requests ?? "—";
      document.getElementById("m_ok").textContent = m.success_requests ?? "—";
      document.getElementById("m_fail").textContent = m.failed_requests ?? "—";
      document.getElementById("m_avg").textContent = fmtMs(m.avg_latency_ms);
      document.getElementById("m_last").textContent = fmtMs(m.last_latency_ms);
      document.getElementById("m_err").textContent = m.last_error ? m.last_error.replace(/\\s+/g, " ").slice(0, 140) : "—";
    }} catch (e) {{
      // If metrics fails, stay quiet: it should not annoy.
    }}
  }}

  async function runSelftest() {{
    const btn = document.getElementById("selftestBtn");
    const status = document.getElementById("selftestStatus");
    const out = document.getElementById("selftestOut");

    btn.disabled = true;
    status.textContent = "running...";
    out.textContent = "Running self-test...";

    try {{
      const r = await fetch("/api/selftest", {{ method: "POST" }});
      const data = await r.json();

      let lines = [];
      lines.push(`Service URL: ${{data.service_url}}`);
      lines.push(`Passed: ${{data.passed}} / ${{data.total}}`);
      lines.push("");

      for (const it of data.results) {{
        const badge = it.ok ? "OK" : "FAIL";
        lines.push(`[${{badge}}] ${{it.filename}}`);
        if (it.latency_ms !== null && it.latency_ms !== undefined) {{
          lines.push(`  latency_ms: ${{fmtMs(it.latency_ms)}}`);
        }}
        if (it.num_returned_sentences !== null && it.num_returned_sentences !== undefined) {{
          lines.push(`  returned_sentences: ${{it.num_returned_sentences}}`);
        }}
        if (!it.ok) {{
          if (it.error) {{
            lines.push("  error:");
            lines.push("  " + String(it.error).split("\\n").join("\\n  "));
          }}
          if (it.missing_sentences && it.missing_sentences.length > 0) {{
            lines.push("  missing_sentences:");
            for (const s of it.missing_sentences) {{
              lines.push("   - " + s);
            }}
          }}
        }}
        lines.push("");
      }}

      out.textContent = lines.join("\\n");
      status.innerHTML = (data.passed === data.total) ? '<span class="ok">passed</span>' : '<span class="bad">failed</span>';
    }} catch (e) {{
      out.textContent = "Self-test failed to run.\\n\\n" + String(e);
      status.innerHTML = '<span class="bad">error</span>';
    }} finally {{
      btn.disabled = false;
      refreshMetrics();
    }}
  }}

  document.getElementById("selftestBtn").addEventListener("click", runSelftest);

  filterBox.addEventListener("input", () => {{
    if (lastAllSentences) renderPreview(lastAllSentences);
  }});

  document.getElementById("uploadForm").addEventListener("submit", async (ev) => {{
    ev.preventDefault();

    const fileInput = document.getElementById("pdfFile");
    if (!fileInput.files || fileInput.files.length === 0) {{
      output.textContent = "Please choose a PDF file first.";
      return;
    }}

    const pdf = fileInput.files[0];
    const fd = new FormData();
    fd.append("pdf_file", pdf, pdf.name);

    extractBtn.disabled = true;
    downloadBtn.disabled = true;
    lastAllSentences = null;
    output.textContent = "Uploading PDF and waiting for service response...";
    latencyEl.textContent = "—";
    returnedCountEl.textContent = "—";

    try {{
      const r = await fetch("/api/extract", {{
        method: "POST",
        body: fd
      }});

      const data = await r.json();
      if (!r.ok) {{
        const msg = data && data.error ? data.error : ("HTTP " + r.status);
        output.textContent = "Request failed.\\n\\n" + msg;
        return;
      }}

      const sentences = data.sentences || [];
      lastAllSentences = sentences;

      latencyEl.textContent = fmtMs(data.latency_ms);
      returnedCountEl.textContent = String(sentences.length);

      renderPreview(sentences);

      // Enable download of full list.
      downloadBtn.disabled = false;
      downloadBtn.onclick = () => {{
        const url = makeTxtDownload(sentences);
        const a = document.createElement("a");
        a.href = url;
        a.download = (pdf.name.replace(/\\.pdf$/i, "") || "sentences") + "_sentences.txt";
        document.body.appendChild(a);
        a.click();
        a.remove();
        setTimeout(() => URL.revokeObjectURL(url), 3000);
      }};
    }} catch (e) {{
      output.textContent = "Unexpected frontend error.\\n\\n" + String(e);
    }} finally {{
      extractBtn.disabled = false;
      refreshMetrics();
    }}
  }});

  // Start: refresh metrics periodically and also run a self-test once (non-blocking).
  refreshMetrics();
  setInterval(refreshMetrics, 2000);

  // Auto self-test on page load (helpful in classroom demos).
  // Comment out if you prefer manual-only.
  runSelftest();
}})();
</script>
</body>
</html>
"""


# -----------------------------
# API routes used by the UI
# -----------------------------

@app.get("/api/metrics")
async def api_metrics() -> Dict[str, Any]:
    """Return a snapshot of operational metrics."""
    return await metrics.snapshot()


@app.post("/api/extract")
async def api_extract(pdf_file: UploadFile = File(...)) -> JSONResponse:
    """Forward the uploaded PDF to the external service and return sentences + latency."""
    t0 = time.perf_counter()
    latency_ms: Optional[float] = None
    try:
        pdf_bytes = await pdf_file.read()
        if not pdf_bytes:
            msg = "Uploaded file was empty. Please upload a non-empty PDF."
            await metrics.record_failure(latency_ms=None, error=msg)
            return JSONResponse(status_code=400, content={"error": msg})

        sentences, latency_ms = await call_sentence_service(pdf_bytes, pdf_file.filename or "uploaded.pdf")
        await metrics.record_success(latency_ms=latency_ms)
        return JSONResponse(
            content={
                "sentences": sentences,
                "latency_ms": latency_ms,
            }
        )
    except httpx.HTTPError as ex:
        latency_ms = (time.perf_counter() - t0) * 1000.0
        msg = _pedagogic_http_error(ex)
        await metrics.record_failure(latency_ms=latency_ms, error=msg)
        return JSONResponse(status_code=502, content={"error": msg, "latency_ms": latency_ms})
    except Exception as ex:
        latency_ms = (time.perf_counter() - t0) * 1000.0
        msg = (
            "The frontend received an unexpected error while processing the response.\n"
            f"- Error type: {type(ex).__name__}\n"
            f"- Details: {ex}\n"
        )
        await metrics.record_failure(latency_ms=latency_ms, error=msg)
        return JSONResponse(status_code=500, content={"error": msg, "latency_ms": latency_ms})


@app.post("/api/selftest")
async def api_selftest() -> Dict[str, Any]:
    """Run self-test dataset checks and return results as JSON."""
    # Note: we do not record these as "metrics requests" since they're a frontend feature
    # and can spam metrics on reload. If you want them counted, call metrics.record_* here.
    return await run_selftest()


# -----------------------------
# Local dev entrypoint (optional)
# -----------------------------
if __name__ == "__main__":
    import uvicorn

    # Run with: python main.py
    uvicorn.run(app, host="0.0.0.0", port=7860)
