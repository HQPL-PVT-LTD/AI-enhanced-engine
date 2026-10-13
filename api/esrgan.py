"""Real-ESRGAN ONNX inference with onnxruntime + numpy.

MODEL=quality (default): RealESRGAN_x4plus, best detail for photos, slower.
MODEL=fast: realesr-general-x4v3, 5MB, much faster but smoother.
"""
import os

import numpy as np
import onnxruntime as ort
from PIL import Image

_MODELS = {"quality": "RealESRGAN_x4plus.onnx", "fast": "realesr-general-x4v3.onnx"}
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models",
                          _MODELS[os.getenv("MODEL", "quality")])
NATIVE = 4  # the network upscales 4x; 2x is produced by 4x then a Lanczos downscale


class Upscaler:
    def __init__(self, tile: int = 256, threads: int = 0):
        opts = ort.SessionOptions()
        if threads:
            opts.intra_op_num_threads = threads
        if hasattr(ort, "preload_dlls"):  # lets onnxruntime-gpu find the pip-installed CUDA libs
            try:
                ort.preload_dlls()
            except Exception:
                pass
        providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider")
                     if p in ort.get_available_providers()]
        self.sess = ort.InferenceSession(MODEL_PATH, opts, providers=providers)
        self.tile = tile

    def _x4(self, img: Image.Image) -> np.ndarray:
        x = np.asarray(img.convert("RGB"), dtype=np.float32).transpose(2, 0, 1)[None] / 255.0
        _, _, h, w = x.shape
        s, t, pad = NATIVE, self.tile, 16
        out = np.empty((1, 3, h * s, w * s), dtype=np.float32)
        for y0 in range(0, h, t):
            for x0 in range(0, w, t):
                y1, x1 = min(y0 + t, h), min(x0 + t, w)
                py0, px0 = max(y0 - pad, 0), max(x0 - pad, 0)
                py1, px1 = min(y1 + pad, h), min(x1 + pad, w)
                res = self.sess.run(None, {"input": np.ascontiguousarray(x[:, :, py0:py1, px0:px1])})[0]
                oy, ox = (y0 - py0) * s, (x0 - px0) * s
                out[:, :, y0 * s:y1 * s, x0 * s:x1 * s] = res[:, :, oy:oy + (y1 - y0) * s,
                                                              ox:ox + (x1 - x0) * s]
        return out[0].transpose(1, 2, 0)

    def __call__(self, img: Image.Image, scale: int) -> Image.Image:
        arr = (self._x4(img).clip(0, 1) * 255.0).round().astype(np.uint8)
        out = Image.fromarray(arr)
        if scale != NATIVE:
            out = out.resize((img.width * scale, img.height * scale), Image.LANCZOS)
        return out

    def to_long_edge(self, img: Image.Image, long_edge: int) -> Image.Image:
        """Output whose longer side is exactly long_edge px (aspect ratio kept)."""
        k = long_edge / max(img.size)
        size = (max(1, round(img.width * k)), max(1, round(img.height * k)))
        if k <= 1:  # already large enough: no need to run the network
            return img.resize(size, Image.LANCZOS)
        arr = (self._x4(img).clip(0, 1) * 255.0).round().astype(np.uint8)
        return Image.fromarray(arr).resize(size, Image.LANCZOS)
