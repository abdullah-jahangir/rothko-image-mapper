"""
color_extractor.py
Extracts a dominant color palette from an image using K-means clustering in LAB color space.
"""

from __future__ import annotations

import hashlib
import io
from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image
from skimage import color as skcolor
from sklearn.cluster import KMeans


def _load_image(source: Union[str, Path, bytes, io.IOBase]) -> Image.Image:
    """Open an image from a file path, bytes, or file-like object."""
    if isinstance(source, (str, Path)):
        img = Image.open(source)
    elif isinstance(source, bytes):
        img = Image.open(io.BytesIO(source))
    else:
        img = Image.open(source)
    return img.convert("RGB")


def _resize_for_speed(img: Image.Image, max_side: int = 200) -> Image.Image:
    """Resize so the longest side is at most max_side px (preserves aspect ratio)."""
    w, h = img.size
    if max(w, h) <= max_side:
        return img
    scale = max_side / max(w, h)
    new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
    return img.resize(new_size, Image.LANCZOS)


def extract_palette(
    source: Union[str, Path, bytes, io.IOBase],
    n_clusters: int = 6,
) -> list[dict]:
    """
    Extract dominant colors from an image.

    Parameters
    ----------
    source : path, bytes, or file-like object
    n_clusters : number of dominant colors to extract (default 6)

    Returns
    -------
    List of dicts sorted by weight (descending):
        [{"lab": [L, a, b], "rgb": [R, G, B], "weight": float}, ...]
    """
    img = _load_image(source)
    img = _resize_for_speed(img, max_side=200)

    # Pixel array as float [0, 1] for skimage
    pixels_rgb = np.array(img, dtype=np.float32) / 255.0  # (H, W, 3)
    pixels_lab = skcolor.rgb2lab(pixels_rgb)               # (H, W, 3) in LAB
    flat_lab = pixels_lab.reshape(-1, 3)                   # (N, 3)

    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    labels = kmeans.fit_predict(flat_lab)
    centers_lab = kmeans.cluster_centers_  # (n_clusters, 3)

    # Compute per-cluster pixel weights
    total = len(labels)
    weights = np.array(
        [np.sum(labels == i) / total for i in range(n_clusters)],
        dtype=np.float32,
    )

    # Convert cluster centers back to RGB
    centers_lab_img = centers_lab.reshape(1, n_clusters, 3)
    centers_rgb_float = skcolor.lab2rgb(centers_lab_img).reshape(n_clusters, 3)
    centers_rgb = np.clip(centers_rgb_float * 255, 0, 255).astype(np.uint8)

    # Build and sort result by weight descending
    palette = [
        {
            "lab": centers_lab[i].tolist(),
            "rgb": centers_rgb[i].tolist(),
            "weight": float(weights[i]),
        }
        for i in range(n_clusters)
    ]
    palette.sort(key=lambda c: c["weight"], reverse=True)
    return palette


def palette_from_bytes(data: bytes, n_clusters: int = 6) -> list[dict]:
    """Convenience wrapper for Streamlit uploaded file bytes."""
    return extract_palette(data, n_clusters=n_clusters)


def file_hash(data: bytes) -> str:
    """MD5 hash of raw bytes — used as Streamlit cache key."""
    return hashlib.md5(data).hexdigest()
