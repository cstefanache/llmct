"""Render numpy matrices to PNG bytes using matplotlib colormaps (no axes)."""
from __future__ import annotations

import io

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import cm  # noqa: E402
from PIL import Image  # noqa: E402


def matrix_to_png(
    mat: np.ndarray,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    signed: bool = False,
) -> bytes:
    """Render a 2D matrix as PNG — one value per pixel, no interpolation/axes."""
    m = np.asarray(mat, dtype=np.float32)
    if m.ndim != 2:
        raise ValueError(f"expected 2D, got {m.shape}")
    if signed:
        vabs = float(np.nanmax(np.abs(m))) or 1.0
        vmin = -vabs if vmin is None else vmin
        vmax = vabs if vmax is None else vmax
    else:
        vmin = float(np.nanmin(m)) if vmin is None else vmin
        vmax = float(np.nanmax(m)) if vmax is None else vmax
    if vmax - vmin < 1e-12:
        vmax = vmin + 1e-12
    norm = (m - vmin) / (vmax - vmin)
    norm = np.clip(norm, 0.0, 1.0)
    rgba = (cm.get_cmap(cmap)(norm) * 255).astype(np.uint8)
    img = Image.fromarray(rgba, mode="RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=False)
    return buf.getvalue()
