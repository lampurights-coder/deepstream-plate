"""FastAPI control plane: get-config / put-config / start / stop / results / health.

Config is YAML on disk (default ``cameras.yaml`` next to this package). Changes while
the pipeline is running are rejected — stop, PUT /config, then POST /start.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from .minio_sink import MinioHitSink
from .pipeline import cameras_from_config, pipeline_runner

logger = logging.getLogger("ds_lpr.app")

_PKG_DIR = Path(__file__).resolve().parent
_DEFAULT_CONFIG = _PKG_DIR / "cameras.yaml"
CONFIG_PATH = Path(os.environ.get("DS_LPR_CONFIG", str(_DEFAULT_CONFIG))).resolve()

# Default /docs pulls Swagger UI from a CDN; that often shows a blank page offline.
# Serve a self-contained docs page instead (only needs /openapi.json).
app = FastAPI(
    title="ds-lpr-service",
    version="0.1.0",
    docs_url=None,
    redoc_url=None,
)

_LOCAL_DOCS_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>ds-lpr-service API</title>
<style>
  :root { color-scheme: light; --bg:#f4f6f8; --card:#fff; --ink:#15202b; --muted:#5b6b7c; --line:#d7dee7; --acc:#0b6bcb; --ok:#0a7a3e; --bad:#b42318; }
  * { box-sizing: border-box; }
  body { margin:0; font:14px/1.45 system-ui,Segoe UI,sans-serif; background:var(--bg); color:var(--ink); }
  header { padding:20px 24px; background:linear-gradient(120deg,#0b6bcb,#184e8d); color:#fff; }
  header h1 { margin:0 0 6px; font-size:22px; font-weight:650; }
  header p { margin:0; opacity:.9; }
  main { max-width:980px; margin:0 auto; padding:20px 16px 48px; }
  .meta { display:flex; gap:12px; flex-wrap:wrap; margin:14px 0 18px; color:var(--muted); }
  .meta a { color:var(--acc); }
  .ep { background:var(--card); border:1px solid var(--line); border-radius:10px; margin:0 0 12px; overflow:hidden; }
  .ep-h { display:flex; gap:10px; align-items:center; padding:12px 14px; cursor:pointer; }
  .ep-h:hover { background:#f8fafc; }
  .method { font:700 11px/1 ui-monospace,monospace; padding:5px 8px; border-radius:6px; color:#fff; min-width:56px; text-align:center; }
  .GET{background:#0a7a3e} .POST{background:#0b6bcb} .PUT{background:#9a3412} .DELETE{background:#b42318}
  .path { font:600 13px/1.2 ui-monospace,monospace; }
  .sum { color:var(--muted); margin-left:auto; }
  .body { display:none; padding:0 14px 14px; border-top:1px solid var(--line); }
  .ep.open .body { display:block; }
  label { display:block; margin:10px 0 4px; color:var(--muted); font-size:12px; }
  input, textarea { width:100%; padding:8px 10px; border:1px solid var(--line); border-radius:8px; font:13px ui-monospace,monospace; }
  textarea { min-height:120px; resize:vertical; }
  .row { display:flex; gap:8px; margin-top:10px; flex-wrap:wrap; }
  button { border:0; border-radius:8px; padding:8px 14px; background:var(--acc); color:#fff; font-weight:600; cursor:pointer; }
  button.secondary { background:#64748b; }
  pre { margin:10px 0 0; padding:12px; background:#0f172a; color:#e2e8f0; border-radius:8px; overflow:auto; white-space:pre-wrap; word-break:break-word; }
  .status { font-weight:700; }
  .status.ok { color:var(--ok); } .status.bad { color:var(--bad); }
</style>
</head>
<body>
<header>
  <h1>ds-lpr-service</h1>
  <p>DeepStream LPR control plane — local docs (no CDN)</p>
</header>
<main>
  <div class="meta">
    <span id="title">Loading OpenAPI…</span>
    <a href="/openapi.json">openapi.json</a>
    <a href="/health">/health</a>
  </div>
  <div id="list"></div>
</main>
<script>
async function load() {
  const spec = await fetch('/openapi.json').then(r => r.json());
  document.getElementById('title').textContent = (spec.info && spec.info.title || 'API') + ' v' + ((spec.info && spec.info.version) || '');
  const list = document.getElementById('list');
  const paths = spec.paths || {};
  const methods = ['get','post','put','delete','patch'];
  for (const path of Object.keys(paths).sort()) {
    for (const m of methods) {
      const op = paths[path][m];
      if (!op) continue;
      const el = document.createElement('section');
      el.className = 'ep';
      const sum = op.summary || op.operationId || '';
      el.innerHTML = `
        <div class="ep-h"><span class="method ${m.toUpperCase()}">${m.toUpperCase()}</span>
        <span class="path">${path}</span><span class="sum">${sum}</span></div>
        <div class="body">
          <p>${op.description || ''}</p>
          <div class="params"></div>
          <label>JSON body (optional)</label>
          <textarea class="body-input" placeholder='{}'></textarea>
          <div class="row">
            <button type="button" class="go">Try it</button>
            <button type="button" class="secondary clear">Clear</button>
            <span class="status"></span>
          </div>
          <pre class="out">(response)</pre>
        </div>`;
      const body = el.querySelector('.body');
      const paramsBox = el.querySelector('.params');
      const queryParams = (op.parameters || []).filter(p => p.in === 'query');
      for (const p of queryParams) {
        const lab = document.createElement('label');
        lab.textContent = p.name + (p.required ? ' *' : '');
        const inp = document.createElement('input');
        inp.dataset.param = p.name;
        inp.placeholder = (p.schema && (p.schema.type || '')) || 'string';
        if (p.name === 'limit') inp.value = '50';
        paramsBox.appendChild(lab);
        paramsBox.appendChild(inp);
      }
      el.querySelector('.ep-h').onclick = () => el.classList.toggle('open');
      el.querySelector('.clear').onclick = (ev) => {
        ev.stopPropagation();
        el.querySelector('.out').textContent = '(response)';
        el.querySelector('.status').textContent = '';
      };
      el.querySelector('.go').onclick = async (ev) => {
        ev.stopPropagation();
        const qs = new URLSearchParams();
        for (const inp of paramsBox.querySelectorAll('input[data-param]')) {
          if (inp.value !== '') qs.set(inp.dataset.param, inp.value);
        }
        const url = path + (qs.toString() ? ('?' + qs.toString()) : '');
        const opts = { method: m.toUpperCase(), headers: {} };
        const raw = el.querySelector('.body-input').value.trim();
        if (raw && ['POST','PUT','PATCH'].includes(opts.method)) {
          opts.headers['Content-Type'] = 'application/json';
          opts.body = raw;
        }
        const statusEl = el.querySelector('.status');
        const out = el.querySelector('.out');
        statusEl.textContent = '…';
        statusEl.className = 'status';
        try {
          const res = await fetch(url, opts);
          const text = await res.text();
          let pretty = text;
          try { pretty = JSON.stringify(JSON.parse(text), null, 2); } catch (_) {}
          statusEl.textContent = res.status + ' ' + res.statusText;
          statusEl.className = 'status ' + (res.ok ? 'ok' : 'bad');
          out.textContent = pretty || '(empty)';
        } catch (err) {
          statusEl.textContent = 'error';
          statusEl.className = 'status bad';
          out.textContent = String(err);
        }
      };
      list.appendChild(el);
    }
  }
}
load().catch(err => {
  document.getElementById('title').textContent = 'Failed to load /openapi.json: ' + err;
});
</script>
</body>
</html>
"""

# In-memory mirror of the YAML file (reloaded on start / after PUT).
_yaml_config: Dict[str, Any] = {}
_uploader: Optional[MinioHitSink] = None


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("config must be a top-level YAML mapping")
    return data


def _save_yaml(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)


def _ensure_config_loaded() -> Dict[str, Any]:
    global _yaml_config
    if not _yaml_config:
        try:
            _yaml_config = _load_yaml(CONFIG_PATH)
        except Exception as exc:
            logger.warning("initial config load failed: %s", exc)
            _yaml_config = {}
    return _yaml_config


@app.on_event("startup")
def _startup() -> None:
    global _yaml_config
    try:
        _yaml_config = _load_yaml(CONFIG_PATH)
        logger.info("Loaded config from %s (%d cameras)", CONFIG_PATH, len(_yaml_config.get("cameras") or []))
    except Exception as exc:
        logger.warning("Could not load config at startup: %s", exc)
        _yaml_config = {}


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse(url="/docs")


@app.get("/docs", include_in_schema=False, response_class=HTMLResponse)
def local_docs() -> str:
    """Offline-friendly API docs (no external CDN)."""
    return _LOCAL_DOCS_HTML


@app.get("/health")
def health() -> dict:
    """``{running, cameras, last_error}`` (+ OCR/frame counters from the runner)."""
    return pipeline_runner.health()


@app.get("/get-config")
def get_config() -> dict:
    """Current YAML config plus ``running`` flag."""
    cfg = _ensure_config_loaded()
    return {
        "running": pipeline_runner.running,
        "config_path": str(CONFIG_PATH),
        "config": cfg,
    }


@app.put("/config")
async def put_config(request: Request) -> dict:
    """Replace YAML on disk. Rejected while the pipeline is running."""
    global _yaml_config

    if pipeline_runner.running:
        raise HTTPException(
            status_code=400,
            detail="Cannot overwrite config while service is running. Stop first.",
        )

    body_bytes = await request.body()
    if not body_bytes:
        raise HTTPException(status_code=400, detail="Empty request body.")

    body_text = body_bytes.decode("utf-8", errors="ignore")
    parsed: Dict[str, Any]
    try:
        loaded = json.loads(body_text)
        if not isinstance(loaded, dict):
            raise ValueError("JSON must be a top-level object.")
        parsed = loaded
    except Exception:
        try:
            loaded = yaml.safe_load(body_text) or {}
            if not isinstance(loaded, dict):
                raise ValueError("YAML must be a top-level mapping.")
            parsed = loaded
        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to parse body as JSON or YAML: {exc}",
            ) from exc

    # Validate cameras shape early so bad configs never hit disk silently.
    try:
        cameras_from_config(parsed)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        _save_yaml(CONFIG_PATH, parsed)
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Failed to write config file: {exc}"
        ) from exc

    _yaml_config = parsed
    return {
        "status": "ok",
        "message": f"Config written to {CONFIG_PATH}",
        "config": parsed,
    }


@app.post("/start")
def start() -> dict:
    """Build and PLAY the DeepStream pipeline from the current YAML."""
    global _uploader

    if pipeline_runner.running:
        raise HTTPException(status_code=400, detail="Pipeline already running.")

    try:
        cfg = _load_yaml(CONFIG_PATH)
    except Exception as exc:
        raise HTTPException(
            status_code=400, detail=f"Failed to load config: {exc}"
        ) from exc

    global _yaml_config
    _yaml_config = cfg

    try:
        rows = cameras_from_config(cfg)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        sink = MinioHitSink.from_config(cfg.get("minio"))
        sink.ensure_buckets(r.bucket for r in rows)
        _uploader = sink
    except Exception as exc:
        logger.exception("MinIO setup failed")
        raise HTTPException(
            status_code=500, detail=f"MinIO setup failed: {exc}"
        ) from exc

    try:
        result = pipeline_runner.start(cfg, uploader=_uploader)
    except Exception as exc:
        logger.exception("pipeline start failed")
        raise HTTPException(
            status_code=500, detail=f"Pipeline start failed: {exc}"
        ) from exc

    return result


@app.post("/stop")
def stop() -> dict:
    """EOS + join; idle after return."""
    try:
        return pipeline_runner.stop()
    except Exception as exc:
        logger.exception("pipeline stop failed")
        raise HTTPException(
            status_code=500, detail=f"Pipeline stop failed: {exc}"
        ) from exc


@app.get("/results")
def results(limit: int = Query(50, ge=1, le=500)) -> dict:
    """Recent OCR hits: ``cam_id``, ``ocr``, ``conf``, ``bucket``, ``object_path``, ``ts``."""
    hits = pipeline_runner.get_results(limit=limit)
    slim = [
        {
            "cam_id": h.get("cam_id"),
            "ocr": h.get("ocr"),
            "conf": h.get("conf"),
            "bucket": h.get("bucket"),
            "object_path": h.get("object_path"),
            "ts": h.get("ts"),
            "bbox": h.get("bbox"),
        }
        for h in hits
    ]
    return {"count": len(slim), "results": slim}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "ds_lpr_service.app:app",
        host="0.0.0.0",
        port=int(os.environ.get("DS_LPR_PORT", "8003")),
        reload=False,
        log_level="info",
    )
