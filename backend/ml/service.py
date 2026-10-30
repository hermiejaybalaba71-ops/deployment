"""HTTP entrypoint for the OCR + Random Forest pipeline on Vercel Services.

Runs as an internal FastAPI service (never publicly routed). The Next.js app
calls it over the ML_URL service binding; requests POST the pipeline payload
as JSON and receive the same NDJSON event stream pipeline.py writes to stdout.

    POST /pipeline   {mode, seed, n_trees, n_synth, applicants, ocr_results}
"""

from __future__ import annotations

import asyncio
import importlib
import json
import os
import queue
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from typing import Iterator
from importlib.metadata import PackageNotFoundError, version

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

ML_DIR = Path(__file__).resolve().parent
NDJSON_MEDIA = "application/x-ndjson; charset=utf-8"
NDJSON_HEADERS = {
    "Content-Type": NDJSON_MEDIA,
    "Cache-Control": "no-cache, no-transform",
    "Connection": "keep-alive",
}
_EOF = object()

app = FastAPI(title="scholar-capstone2 ML pipeline", version="1.0.0")
MAX_SHEET_BYTES = 20 * 1024 * 1024
SHEET_EXTENSIONS = {".csv", ".xlsx", ".xlsm", ".xltx"}


def _runtime_python() -> tuple[str, dict[str, str]]:
    """Run the child pipeline with packages installed by Vercel's Python runtime."""
    runtime_env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    site_packages = [
        path
        for root in (Path("/tmp/_vc_deps/lib"), Path("/var/task/.venv/lib"))
        if root.exists()
        for path in root.glob("python*/site-packages")
        if path.is_dir()
    ]
    if site_packages:
        existing = runtime_env.get("PYTHONPATH", "")
        runtime_env["PYTHONPATH"] = os.pathsep.join(
            [*(str(path) for path in site_packages), existing] if existing else [str(path) for path in site_packages]
        )

    virtualenv_python = Path("/tmp/_vc_deps/bin/python")
    return (str(virtualenv_python) if virtualenv_python.exists() else sys.executable), runtime_env


@app.get("/")
def health() -> JSONResponse:
    ocr_runtime: dict[str, str] = {}
    errors: list[str] = []
    for module_name, display_name in (("cv2", "opencv"), ("onnxruntime", "onnxruntime")):
        try:
            module = importlib.import_module(module_name)
        except (ImportError, OSError) as exc:
            errors.append(f"{display_name}: {exc}")
        else:
            ocr_runtime[display_name] = str(getattr(module, "__version__", "unknown"))

    try:
        importlib.import_module("rapidocr_onnxruntime")
    except (ImportError, OSError) as exc:
        errors.append(f"rapidocr: {exc}")
    else:
        try:
            ocr_runtime["rapidocr"] = version("rapidocr-onnxruntime")
        except PackageNotFoundError:
            errors.append("rapidocr: rapidocr-onnxruntime is not installed")

    return JSONResponse(
        {
            "ok": True,
            "service": "ml",
            "python": sys.version.split()[0],
            "ocr_ready": not errors,
            "ocr_runtime": ocr_runtime,
            "ocr_errors": errors,
        }
    )


def _parse_sheet(filename: str, contents: bytes) -> dict:
    safe_name = filename.replace("\\", "/").rsplit("/", 1)[-1]
    suffix = Path(safe_name).suffix.lower()
    if suffix not in SHEET_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Unsupported file type. Use .csv or .xlsx.")
    if len(contents) > MAX_SHEET_BYTES:
        raise HTTPException(status_code=413, detail="File must be 20 MB or smaller.")

    python_bin, runtime_env = _runtime_python()
    with tempfile.TemporaryDirectory(prefix="sheet-") as work_dir:
        upload_path = Path(work_dir) / f"upload{suffix}"
        upload_path.write_bytes(contents)
        try:
            result = subprocess.run(
                [python_bin, str(ML_DIR / "read_sheet.py"), str(upload_path)],
                cwd=str(ML_DIR),
                env=runtime_env,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=45,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise HTTPException(status_code=504, detail="Reading this file took too long.") from exc

    if not result.stdout.strip():
        errors = result.stderr.strip().splitlines()
        raise HTTPException(
            status_code=500,
            detail=errors[-1] if errors else "The sheet reader returned no output.",
        )
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=500, detail="The sheet reader returned invalid JSON.") from exc
    if not isinstance(payload, dict):
        raise HTTPException(status_code=500, detail="The sheet reader returned an invalid response.")
    if result.returncode != 0 or payload.get("ok") is not True:
        raise HTTPException(status_code=400, detail=str(payload.get("error") or "Could not read this file."))
    return payload


@app.post("/parse-sheet")
async def parse_sheet(request: Request) -> JSONResponse:
    filename = request.headers.get("x-file-name", "")
    if not filename:
        raise HTTPException(status_code=400, detail="Missing x-file-name header.")
    contents = await request.body()
    return JSONResponse(await asyncio.to_thread(_parse_sheet, filename, contents))


@app.post("/pipeline")
def pipeline(payload: dict) -> StreamingResponse:
    return StreamingResponse(_stream_pipeline(payload), media_type=NDJSON_MEDIA, headers=NDJSON_HEADERS)


def _stream_pipeline(payload: dict) -> Iterator[str]:
    python_bin, runtime_env = _runtime_python()
    proc = subprocess.Popen(
        [python_bin, str(ML_DIR / "pipeline.py")],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=str(ML_DIR),
        env=runtime_env,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )

    events: "queue.Queue" = queue.Queue(maxsize=1024)

    def pump(stream, wrap=None) -> None:
        try:
            for line in stream:
                line = line.rstrip("\n")
                if not line.strip():
                    continue
                events.put(wrap(line) if wrap else json.dumps({"type": "log", "msg": line}) + "\n"
                           if not line.startswith("{") else line + "\n")
        except (ValueError, OSError):
            pass
        finally:
            events.put(_EOF)

    threading.Thread(target=pump, args=(proc.stdout,), daemon=True).start()
    threading.Thread(
        target=pump,
        args=(proc.stderr, lambda line: json.dumps({"type": "log", "msg": f"[python] {line}", "stream": "stderr"}) + "\n"),
        daemon=True,
    ).start()

    try:
        assert proc.stdin is not None
        proc.stdin.write(json.dumps(payload))
        proc.stdin.close()
    except (BrokenPipeError, ValueError, OSError):
        pass

    eof_count = 0
    while eof_count < 2:
        item = events.get()
        if item is _EOF:
            eof_count += 1
        else:
            yield item

    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()
