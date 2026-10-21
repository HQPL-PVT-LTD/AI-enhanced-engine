"""Qwen-Image 2.1 + detailer LoRA enhancement through a local ComfyUI server.

The graph is qwen_image_2_1_upscale_v1.json with a LoraLoaderModelOnly between the model loader
and the sampler. ComfyUI must be running (COMFY_URL) with the models from comfy_paths.yaml.
"""
import io
import math
import os
import random
import logging
import time
import uuid

import numpy as np
import requests
from PIL import Image

logger = logging.getLogger("upscale-api")

COMFY_URL = os.getenv("COMFY_URL", "http://127.0.0.1:8188")
UNET = os.getenv("QWEN_UNET", "qwen_image_2.1_int8_convrot.safetensors")
CLIP = os.getenv("QWEN_CLIP", "qwen3vl_8b_int8_convrot.safetensors")
VAE = os.getenv("QWEN_VAE", "qwen_image_2.1_vae_bf16.safetensors")
LORA = os.getenv("QWEN_LORA", "elusarcas-qwen2-1-detailer-v1.safetensors")
LORA_STRENGTH = float(os.getenv("QWEN_LORA_STRENGTH", "0.8"))
STEPS = int(os.getenv("QWEN_STEPS", "15"))
# 1.0 = generate from an empty latent (the original workflow; recomposes the picture and can change
# faces). Lower values start from the encoded original, which keeps composition and identity.
DENOISE = float(os.getenv("QWEN_DENOISE", "0.55"))
# Optional detail pass (off by default: at 0.35 it left tiles almost unchanged): the stretched result is re-rendered tile by
# tile at this denoise so every region gets real detail (slow: about one model run per TILE x TILE area).
TILE = int(os.getenv("QWEN_TILE", "1280"))
TILE_OVERLAP = int(os.getenv("QWEN_TILE_OVERLAP", "192"))
TILE_DENOISE = float(os.getenv("QWEN_TILE_DENOISE", "0"))
# Real-ESRGAN (ComfyUI/models/upscale_models) does the last stretch beyond the model's native size.
SR_MODEL = os.getenv("QWEN_SR_MODEL", "RealESRGAN_x4plus.pth")
TIMEOUT = float(os.getenv("QWEN_TIMEOUT", "600"))
PROMPT = ("Enhance this image to high resolution. Keep the content, composition, colors, lighting and style "
          "exactly the same. Only increase sharpness and fine detail. Realistic photograph with natural skin "
          "texture, visible pores, fine hair strands and subtle film grain. Not smooth, not airbrushed, "
          "not plastic, not a painting or cartoon.")

class QwenError(RuntimeError):
    """ComfyUI is unreachable, rejected the workflow, failed, or timed out."""


PIXEL_BUDGET = 2048 * 2048  # same sizing rule as the workflow: up to 4x, at most ~2048x2048 pixels


def working_size(w: int, h: int) -> tuple:
    # never work below the input resolution: shrinking first and stretching back up loses detail
    k = max(1.0, min(4.0, math.sqrt(PIXEL_BUDGET / (w * h))))
    return max(16, round(w * k / 16) * 16), max(16, round(h * k / 16) * 16)


def _graph(image_name: str, size: tuple, seed: int, lora_strength: float, denoise: float) -> dict:
    g = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": UNET, "weight_dtype": "default"}},
        "2": {"class_type": "LoraLoaderModelOnly",
              "inputs": {"model": ["1", 0], "lora_name": LORA, "strength_model": lora_strength}},
        "3": {"class_type": "CLIPLoader", "inputs": {"clip_name": CLIP, "type": "qwen_image", "device": "default"}},
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": VAE}},
        "5": {"class_type": "LoadImage", "inputs": {"image": image_name}},
        "6": {"class_type": "TextEncodeQwenImage21",
              "inputs": {"clip": ["3", 0], "prompt": PROMPT, "negative_prompt": "", "resolution": 1024,
                         "images.image_1": ["5", 0], "vae": ["4", 0]}},
        "7": {"class_type": "EmptyLatentImage", "inputs": {"width": size[0], "height": size[1], "batch_size": 1}},
        "11": {"class_type": "ImageScale", "inputs": {"image": ["5", 0], "upscale_method": "lanczos",
                                                      "width": size[0], "height": size[1], "crop": "disabled"}},
        "12": {"class_type": "VAEEncode", "inputs": {"pixels": ["11", 0], "vae": ["4", 0]}},
        "8": {"class_type": "KSampler",
              "inputs": {"model": ["2", 0], "positive": ["6", 0], "negative": ["6", 1], "latent_image": ["7", 0],
                         "seed": seed, "steps": STEPS, "cfg": 1.0, "sampler_name": "euler",
                         "scheduler": "simple", "denoise": denoise}},
        "9": {"class_type": "VAEDecode", "inputs": {"samples": ["8", 0], "vae": ["4", 0]}},
        "10": {"class_type": "PreviewImage", "inputs": {"images": ["9", 0]}},
    }
    if denoise < 1.0:  # img2img: start from the original instead of an empty latent
        g["8"]["inputs"]["latent_image"] = ["12", 0]
    return g


def _esrgan_graph(image_name: str) -> dict:
    return {
        "1": {"class_type": "LoadImage", "inputs": {"image": image_name}},
        "2": {"class_type": "UpscaleModelLoader", "inputs": {"model_name": SR_MODEL}},
        "3": {"class_type": "ImageUpscaleWithModel", "inputs": {"upscale_model": ["2", 0], "image": ["1", 0]}},
        "10": {"class_type": "PreviewImage", "inputs": {"images": ["3", 0]}},
    }


def _run(img: Image.Image, size: tuple, seed: int, lora_strength: float, denoise: float,
         sr: bool = False) -> Image.Image:
    """One ComfyUI run on img at working size `size`; returns the decoded result at that size.
    sr=True runs the Real-ESRGAN 4x model instead of the Qwen graph (result is 4x the input)."""
    name = f"api_{uuid.uuid4().hex}.png"
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    t0 = time.time()
    try:
        r = requests.post(f"{COMFY_URL}/upload/image", files={"image": (name, buf.getvalue(), "image/png")},
                          timeout=60)
        r.raise_for_status()
        name = r.json()["name"]
        graph = _esrgan_graph(name) if sr else _graph(name, size, seed, lora_strength, denoise)
        r = requests.post(f"{COMFY_URL}/prompt", json={"prompt": graph}, timeout=30)
        if r.status_code != 200:
            raise QwenError(f"ComfyUI rejected the workflow: {r.text[:500]}")
        pid = r.json()["prompt_id"]
        logger.info("comfy prompt %s queued (%dx%d, steps=%d, denoise=%.2f, uploaded in %.1fs)", pid[:8],
                    size[0], size[1], STEPS, denoise, time.time() - t0)
        deadline = time.time() + TIMEOUT
        while True:
            hist = requests.get(f"{COMFY_URL}/history/{pid}", timeout=30).json().get(pid)
            if hist and hist.get("status", {}).get("completed") is not None:
                break
            if time.time() > deadline:
                logger.error("comfy prompt %s timed out after %.0fs, interrupting", pid[:8], TIMEOUT)
                requests.post(f"{COMFY_URL}/interrupt", timeout=10)
                raise QwenError("Qwen enhancement timed out")
            time.sleep(1)
        logger.info("comfy prompt %s finished in %.1fs (%s)", pid[:8], time.time() - t0,
                    hist["status"].get("status_str"))
        if hist["status"].get("status_str") == "error":
            raise QwenError(f"ComfyUI failed: {hist['status'].get('messages', [])[-1:]}")
        out = hist["outputs"]["10"]["images"][0]
        r = requests.get(f"{COMFY_URL}/view", params=out, timeout=60)
        r.raise_for_status()
        res = Image.open(io.BytesIO(r.content)).convert("RGB")
    except requests.ConnectionError:
        logger.error("ComfyUI not reachable at %s", COMFY_URL)
        raise QwenError(f"ComfyUI is not reachable at {COMFY_URL}")
    return res if sr or res.size == size else res.resize(size, Image.LANCZOS)


def _starts(length: int, tile: int, step: int) -> list:
    if length <= tile:
        return [0]
    xs = list(range(0, length - tile, step))
    return xs + [length - tile]


def _ramp(n: int, overlap: int, fade_lo: bool, fade_hi: bool) -> np.ndarray:
    w = np.ones(n, np.float32)
    k = min(overlap, n // 2)
    if k and fade_lo:
        w[:k] = np.linspace(1 / (k + 1), 1, k, dtype=np.float32)
    if k and fade_hi:
        w[n - k:] = np.linspace(1, 1 / (k + 1), k, dtype=np.float32)
    return w


def _refine_tiles(img: Image.Image, seed: int, lora_strength: float) -> Image.Image:
    """Re-render img tile by tile (img2img at TILE_DENOISE) and feather the overlaps together."""
    W, H = img.size
    step = max(64, TILE - TILE_OVERLAP)
    xs, ys = _starts(W, TILE, step), _starts(H, TILE, step)
    acc = np.zeros((H, W, 3), np.float32)
    wsum = np.zeros((H, W, 1), np.float32)
    total, n = len(xs) * len(ys), 0
    for y in ys:
        for x in xs:
            tw, th = min(TILE, W), min(TILE, H)
            tile = img.crop((x, y, x + tw, y + th))
            size = (max(16, round(tw / 16) * 16), max(16, round(th / 16) * 16))
            t0 = time.time()
            out = _run(tile, size, seed, lora_strength, TILE_DENOISE)
            if out.size != tile.size:
                out = out.resize(tile.size, Image.LANCZOS)
            wy = _ramp(th, TILE_OVERLAP, y > 0, y + th < H)
            wx = _ramp(tw, TILE_OVERLAP, x > 0, x + tw < W)
            w = (wy[:, None] * wx[None, :])[..., None]
            acc[y:y + th, x:x + tw] += np.asarray(out, np.float32) * w
            wsum[y:y + th, x:x + tw] += w
            n += 1
            logger.info("detail tile %d/%d done in %.0fs", n, total, time.time() - t0)
    return Image.fromarray(np.clip(acc / wsum + 0.5, 0, 255).astype(np.uint8))


def _enhance_large(img: Image.Image, seed: int, lora_strength: float, denoise: float) -> Image.Image:
    """Already-big image, original size wanted: run the model at ~PIXEL_BUDGET (fast), then apply what it
    changed (colour, tone, detail) to the full-resolution original so no sharpness is lost."""
    k = math.sqrt(PIXEL_BUDGET / (img.width * img.height))
    small = (max(16, round(img.width * k / 16) * 16), max(16, round(img.height * k / 16) * 16))
    base = img.resize(small, Image.LANCZOS)
    res = _run(base, small, seed, lora_strength, denoise)
    delta = np.asarray(res, np.float32) - np.asarray(base, np.float32)
    channels = [Image.fromarray(delta[..., c]).resize(img.size, Image.BICUBIC) for c in range(3)]
    up = np.stack([np.asarray(c, np.float32) for c in channels], axis=-1)
    return Image.fromarray(np.clip(np.asarray(img, np.float32) + up + 0.5, 0, 255).astype(np.uint8))


def enhance(img: Image.Image, out_size: tuple, lora_strength: float = None, seed: int = None,
            denoise: float = None) -> Image.Image:
    """Run the Qwen graph on img and return it at exactly out_size (tiled detail pass when it is larger
    than the model's native working size)."""
    seed = random.randrange(2**32) if seed is None else seed
    lora_strength = LORA_STRENGTH if lora_strength is None else lora_strength
    dn = DENOISE if denoise is None else denoise
    if img.width * img.height > PIXEL_BUDGET * 1.1:
        # already big (e.g. 11 MP): a model run at native size is far slower (attention cost grows faster than
        # the pixel count) and timed out. Enhance at ~PIXEL_BUDGET, then resize to the wanted output.
        res = _enhance_large(img, seed, lora_strength, dn)
        return res if res.size == tuple(out_size) else res.resize(out_size, Image.LANCZOS)
    res = _run(img, working_size(*img.size), seed, lora_strength, dn)
    if res.size == out_size:
        return res
    stretch = max(out_size[0] / res.width, out_size[1] / res.height)
    if stretch > 1.05 and SR_MODEL:  # real super-resolution instead of a blurry Lanczos stretch
        res = _run(res, res.size, seed, lora_strength, dn, sr=True)
    res = res.resize(out_size, Image.LANCZOS)
    if stretch > 1.05 and TILE_DENOISE > 0:
        return _refine_tiles(res, seed, lora_strength)
    return res
