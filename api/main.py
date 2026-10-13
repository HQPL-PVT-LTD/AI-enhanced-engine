"""Image upscale API.

POST /upscale {"post_id": "123", "image_url": "<url>", "scale": 2 | 4}
  -> downloads the image, upscales it with Real-ESRGAN, stores it as
     enhanced_images/<post_id>/<name>_<scale>x.<ext> and returns its path and download_url
     (same response style as the old CodeFormer API).

  pip install -r api/requirements.txt && ./run.sh
"""
import io
import ipaddress
import os
import socket
import logging
import mimetypes
import threading
import time
import uuid
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional
from urllib.parse import unquote, urljoin, urlparse

import requests
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse
from PIL import Image
from pydantic import BaseModel, Field

from . import storage
from .esrgan import Upscaler

logger = logging.getLogger("upscale-api")

MAX_DOWNLOAD_BYTES = int(os.getenv("MAX_DOWNLOAD_BYTES", str(25 * 1024 * 1024)))
MAX_OUTPUT_PIXELS = int(os.getenv("MAX_OUTPUT_PIXELS", str(8192 * 8192)))
TILE = int(os.getenv("TILE", "256"))
# When set, every endpoint requires this value in the X-API-Key header
API_KEY = os.getenv("API_KEY")
# Optional: comma separated hostnames that image_url may point to (empty = any public host)
ALLOWED_IMAGE_HOSTS = {h.strip().lower() for h in os.getenv("ALLOWED_IMAGE_HOSTS", "").split(",") if h.strip()}
# Stored images are deleted this many minutes after they were created (0 = keep forever)
FILE_TTL_MINUTES = float(os.getenv("FILE_TTL_MINUTES", "120"))
MAX_TRACKED_JOBS = 1000

LONG_EDGE = {2: 2048, 4: 3840}  # /enhance: upscale=2 -> 2K, upscale=4 -> 4K (longer side in px)

_upscaler = None
_lock = threading.Lock()  # one inference at a time keeps memory bounded


app = FastAPI(title="Image Upscale API", docs_url=None, redoc_url=None, openapi_url=None,
              version="1.0.0")


class UpscaleRequest(BaseModel):
    post_id: str = Field(..., description="Post identifier, used as the folder name")
    image_url: str = Field(..., description="URL of the image to upscale")
    scale: Literal[2, 4] = Field(2, description="Output scale, 2 or 4, stored as <name>_<scale>x")


class EnhanceRequest(BaseModel):
    post_id: str = Field(..., description="Post identifier, used as the folder name")
    image_url: str = Field(..., description="URL of the image to enhance")
    # Accepted for compatibility with the old CodeFormer API; Real-ESRGAN has no equivalent, so it is ignored
    fidelity_weight: float = Field(0.7, ge=0, le=1)
    upscale: Literal[2, 4] = Field(2, description="2 = 2K output (2048px long side), 4 = 4K output (3840px long side)")


job_status = OrderedDict()  # most recent MAX_TRACKED_JOBS jobs, in memory
job_status_lock = threading.Lock()


def set_job(job_id: str, **fields):
    with job_status_lock:
        job_status.setdefault(job_id, {}).update(fields)
        while len(job_status) > MAX_TRACKED_JOBS:
            job_status.popitem(last=False)


def require_api_key(x_api_key: Optional[str] = Header(None)):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(401, "Invalid or missing API key")


def filename_from_url(url: str) -> str:
    """https://cdn.example.com/uploads/35936/photo_abc.jpg -> photo_abc.jpg"""
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if ALLOWED_IMAGE_HOSTS and host not in ALLOWED_IMAGE_HOSTS:
        raise HTTPException(400, f"Image host '{host}' is not allowed")
    parts = [unquote(p) for p in parsed.path.split("/") if p]
    if not parts:
        raise HTTPException(400, "Invalid image URL: no file path")
    return storage.safe_filename(parts[-1])


def _check_public_host(url: str) -> None:
    p = urlparse(url)
    if p.scheme not in ("http", "https") or not p.hostname:
        raise HTTPException(400, "url must be an http(s) URL")
    try:
        infos = socket.getaddrinfo(p.hostname, p.port or (443 if p.scheme == "https" else 80))
    except socket.gaierror:
        raise HTTPException(400, "could not resolve url host")
    for info in infos:
        if not ipaddress.ip_address(info[4][0]).is_global:
            raise HTTPException(400, "url must point to a public host")


def download_image(url: str) -> Image.Image:
    # follow redirects manually so every hop is checked against private ranges
    for _ in range(5):
        _check_public_host(url)
        try:
            r = requests.get(url, stream=True, timeout=30, allow_redirects=False,
                             headers={"User-Agent": "upscale-api/1.0"})
        except requests.RequestException as e:
            raise HTTPException(400, f"failed to download image: {e}")
        if r.is_redirect:
            url = urljoin(url, r.headers.get("location", ""))
            r.close()
            continue
        break
    else:
        raise HTTPException(400, "too many redirects")
    if r.status_code != 200:
        raise HTTPException(400, f"download failed with status {r.status_code}")
    data = bytearray()
    for chunk in r.iter_content(1 << 16):
        data += chunk
        if len(data) > MAX_DOWNLOAD_BYTES:
            raise HTTPException(413, "image too large")
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except Exception:
        raise HTTPException(400, "url did not return a valid image")
    return img.convert("RGB")


@app.on_event("startup")
def start_cleanup():
    def loop():
        while True:
            try:
                n = storage.delete_expired_images(FILE_TTL_MINUTES * 60)
                if n:
                    logger.info("Deleted %d expired image(s)", n)
            except Exception:
                logger.exception("expired image cleanup failed")
            time.sleep(30)
    if FILE_TTL_MINUTES > 0:
        threading.Thread(target=loop, daemon=True).start()


@app.get("/")
def root():
    return {
        "message": "Image Upscale API", "version": "1.0.0", "status": "running",
        "endpoints": {
            "/enhance": "POST - Same as /upscale with fields post_id, image_url, fidelity_weight, upscale",
            "/upscale": "POST - Upscale an image from a URL (2x or 4x), returns the stored file path",
            "/files/{post_id}/{filename}": "GET - View a stored upscaled image, DELETE - remove it",
            "/job/{job_id}": "GET - Check job status",
            "/health": "GET - Health check",
        },
    }


@app.post("/upscale", dependencies=[Depends(require_api_key)])
def upscale(req: UpscaleRequest, http_request: Request):
    """Download the image, upscale it 2x/4x, store it as <post_id>/<name>_<scale>x.<ext>"""
    return process(req.post_id, req.image_url, req.scale, http_request)


@app.post("/enhance", dependencies=[Depends(require_api_key)])
def enhance(req: EnhanceRequest, http_request: Request):
    """Same as /upscale, with the old CodeFormer request format (post_id, image_url, fidelity_weight, upscale)"""
    r = process(req.post_id, req.image_url, req.upscale, http_request, long_edge=LONG_EDGE[req.upscale])
    keys = ("job_id", "post_id", "status", "message", "original_url", "file_path", "filename",
            "size_bytes", "download_url")
    return {k: r[k] for k in keys} | {"message": "Image enhancement completed successfully"}


def process(post_id: str, image_url: str, scale: int, http_request: Request,
            long_edge: Optional[int] = None):
    if not storage.is_valid_post_id(post_id):
        raise HTTPException(400, "Invalid post_id")
    filename = filename_from_url(image_url)

    job_id = uuid.uuid4().hex
    set_job(job_id, status="downloading", post_id=post_id,
            created_at=datetime.now().isoformat(), original_url=image_url)
    try:
        img = download_image(image_url)
        k = long_edge / max(img.size) if long_edge else scale
        out_w, out_h = round(img.width * k), round(img.height * k)
        if out_w * out_h > MAX_OUTPUT_PIXELS:
            raise HTTPException(400, f"output {out_w}x{out_h} exceeds the allowed size; "
                                     "use a smaller image or scale 2")
        set_job(job_id, status="enhancing")
        with _lock:
            global _upscaler
            if _upscaler is None:
                _upscaler = Upscaler(TILE)
            result = _upscaler.to_long_edge(img, long_edge) if long_edge else _upscaler(img, scale)

        set_job(job_id, status="saving")
        stored = storage.save_image(result, post_id, filename, scale, label=f"{scale}k" if long_edge else None)
        download_url = f"{str(http_request.base_url).rstrip('/')}/files/{post_id}/{stored['filename']}"
        set_job(job_id, status="completed", file_path=stored["file_path"],
                download_url=download_url, completed_at=datetime.now().isoformat())
        return {
            "job_id": job_id,
            "post_id": post_id,
            "status": "completed",
            "message": "Image upscaling completed successfully",
            "original_url": image_url,
            "scale": stored["scale"],
            "width": stored["width"],
            "height": stored["height"],
            "file_path": stored["file_path"],
            "filename": stored["filename"],
            "size_bytes": stored["size_bytes"],
            "download_url": download_url,
            "expires_in_minutes": FILE_TTL_MINUTES,
        }
    except HTTPException as e:
        set_job(job_id, status="failed", error=e.detail)
        raise
    except Exception as e:
        logger.exception("Job %s failed", job_id)
        set_job(job_id, status="failed", error=str(e))
        raise HTTPException(500, "Error processing request")


@app.get("/files/{post_id}/{filename}", dependencies=[Depends(require_api_key)])
def get_file(post_id: str, filename: str):
    """Show a stored image (displayed in the browser, not downloaded)"""
    path = storage.get_image_path(post_id, filename)
    if path is None:
        raise HTTPException(404, "File not found")
    media_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
    return FileResponse(path, media_type=media_type, filename=path.name,
                        content_disposition_type="inline")


@app.delete("/files/{post_id}/{filename}", dependencies=[Depends(require_api_key)])
def delete_file(post_id: str, filename: str):
    if not storage.delete_image(post_id, filename):
        raise HTTPException(404, "File not found")
    return {"message": "File deleted successfully"}


@app.get("/job/{job_id}", dependencies=[Depends(require_api_key)])
def get_job_status(job_id: str):
    with job_status_lock:
        job = job_status.get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    return job


@app.get("/health")
def health():
    return {"status": "healthy", "output_dir": str(storage.OUTPUT_DIR),
            "output_dir_writable": os.access(str(storage.OUTPUT_DIR), os.W_OK),
            "file_ttl_minutes": FILE_TTL_MINUTES, "auth_enabled": bool(API_KEY)}
