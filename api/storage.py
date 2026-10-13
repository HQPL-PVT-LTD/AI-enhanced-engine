"""Local storage for upscaled images: OUTPUT_DIR/<post_id>/<name>_<scale>x.<ext>"""
import os
import re
import time
from pathlib import Path
from typing import Optional

from PIL import Image

OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", Path(__file__).resolve().parent.parent / "enhanced_images")).resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

_POST_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def is_valid_post_id(post_id: str) -> bool:
    """Post ids become folder names, so only plain characters are allowed"""
    return bool(_POST_ID.match(post_id))


def safe_filename(filename: str) -> str:
    name = re.sub(r"[^A-Za-z0-9._-]", "_", Path(filename).name).lstrip(".")
    return name or "image.png"


def save_image(img: Image.Image, post_id: str, filename: str, scale, label: Optional[str] = None) -> dict:
    label = label or f"{scale}x"
    filename = safe_filename(filename)
    suffix = Path(filename).suffix.lower()
    if suffix not in (".jpg", ".jpeg", ".png"):
        suffix = ".png"
    post_dir = OUTPUT_DIR / post_id
    post_dir.mkdir(parents=True, exist_ok=True)
    path = post_dir / f"{Path(filename).stem}_{label}{suffix}"
    if suffix == ".png":
        img.save(path, "PNG")
    else:
        img.save(path, "JPEG", quality=95)  # keep the bytes in line with the .jpg name
    return {"scale": label, "file_path": str(path), "filename": path.name,
            "width": img.width, "height": img.height, "size_bytes": path.stat().st_size}


def get_image_path(post_id: str, filename: str) -> Optional[Path]:
    if not is_valid_post_id(post_id):
        return None
    path = OUTPUT_DIR / post_id / safe_filename(filename)
    return path if path.is_file() else None


def delete_image(post_id: str, filename: str) -> bool:
    path = get_image_path(post_id, filename)
    if path is None:
        return False
    path.unlink()
    try:
        path.parent.rmdir()
    except OSError:
        pass  # other images remain for this post
    return True


def delete_expired_images(max_age_seconds: float) -> int:
    cutoff = time.time() - max_age_seconds
    deleted = 0
    for post_dir in OUTPUT_DIR.iterdir():
        if not post_dir.is_dir():
            continue
        for path in post_dir.iterdir():
            try:
                if path.is_file() and path.stat().st_mtime < cutoff:
                    path.unlink()
                    deleted += 1
            except OSError:
                pass
        try:
            post_dir.rmdir()
        except OSError:
            pass
    return deleted
