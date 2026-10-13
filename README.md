# AI-enhanced-engine

## Upscale API

API only, no UI, no PyTorch. Real-ESRGAN via onnxruntime (~200MB install). Default `MODEL=quality` (RealESRGAN_x4plus, best photo detail, ~5s for a small image on CPU); `MODEL=fast` uses a 5MB model that is ~3x faster but smoother.

```
pip install -r api/requirements.txt
GPU=1 ./run.sh   # NVIDIA GPU (fast). Plain ./run.sh = CPU. Port 8000, override with PORT=...

curl -X POST localhost:8000/upscale -H 'Content-Type: application/json' \
  -d '{"post_id": "123", "image_url": "https://example.com/a.jpg", "scale": 2}'
```

Same style as the old CodeFormer API: the result is stored as `enhanced_images/<post_id>/<name>_<scale>x.<ext>` and the response returns it:

```json
{"job_id": "...", "post_id": "123", "status": "completed", "scale": "2x", "width": 478, "height": 356,
 "file_path": "/.../enhanced_images/123/a_2x.jpg", "filename": "a_2x.jpg", "size_bytes": 123456,
 "download_url": "http://localhost:8000/files/123/a_2x.jpg", "expires_in_minutes": 120.0}
```

| Endpoint | |
|---|---|
| `POST /upscale` | `post_id`, `image_url`, `scale` (2 or 4) |
| `GET /files/{post_id}/{filename}` | view the stored image; `DELETE` removes it |
| `GET /job/{job_id}` | job status |
| `GET /health` | health check |

Env: `API_KEY` (clients then send `X-API-Key`), `FILE_TTL_MINUTES` (default 120, i.e. 2 hours, 0 = keep), `OUTPUT_DIR`, `ALLOWED_IMAGE_HOSTS`, `MODEL`, `TILE`, `MAX_OUTPUT_PIXELS`, `MAX_DOWNLOAD_BYTES`.
The ONNX files were made with `tools/export_onnx.py`.
