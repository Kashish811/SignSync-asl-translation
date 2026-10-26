"""SignSync web API: serves the frontend, precomputed examples, headline statistics and live translation."""
import json
import os
import tempfile
import threading

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

import engine

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
MAX_BYTES = 30 * 1024 * 1024                      # 30 MB upload limit
ALLOWED = {".mp4", ".mov", ".webm", ".avi", ".mkv"}
lock = threading.Lock()                           # one translation at a time (model is heavy)

app = FastAPI(title="SignSync", docs_url=None, redoc_url=None)


def read_json(name, default):
    path = os.path.join(DATA, name)
    if not os.path.exists(path):
        return default
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@app.get("/api/health")
def health():
    live, reason = engine.status()
    return {"live": live, "reason": reason}


@app.get("/api/stats")
def stats():
    return read_json("stats.json", {})


@app.get("/api/examples")
def examples():
    return read_json("examples.json", [])


@app.post("/api/translate")
async def translate(file: UploadFile = File(...)):
    live, reason = engine.status()
    if not live:
        raise HTTPException(503, f"Live translation is not available: {reason}")
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED:
        raise HTTPException(400, "Please upload a video (mp4, mov, webm, avi or mkv).")
    data = await file.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "Video is larger than 30 MB.")
    if not lock.acquire(blocking=False):
        raise HTTPException(429, "The model is busy with another request. Try again in a minute.")
    path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp.write(data)
            path = tmp.name
        sentence, seconds = engine.translate(path)
        return JSONResponse({"translation": sentence, "seconds": seconds})
    except Exception as exc:                      # surface a readable message to the page
        raise HTTPException(500, str(exc)[:300])
    finally:
        lock.release()
        if path and os.path.exists(path):         # videos are never kept
            os.remove(path)


app.mount("/", StaticFiles(directory=os.path.join(HERE, "static"), html=True), name="static")
