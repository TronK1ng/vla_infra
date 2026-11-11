"""PNG encoding for raw color/depth frames (M2).

Pillow is used because it preserves 16-bit depth PNGs and writes color channels
without OpenCV's BGR ambiguity.  Encoding happens only on the recorder thread.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from ..errors import ErrorCode, VlaInfraError

try:  # pragma: no cover - import outcome is environment-dependent
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None  # type: ignore[assignment]


class ImageCodecError(VlaInfraError):
    code = ErrorCode.DEVICE_FAULT


def _require_pillow() -> None:
    if Image is None:  # pragma: no cover - exercised only without Pillow
        raise ImageCodecError(
            "Pillow is required to record images; install the declared dependency",
            code=ErrorCode.CONFIG_INVALID,
        )


def write_png(path: Path, array: np.ndarray, *, encoding: str) -> None:
    """Write an RGB8 or depth16 array as PNG (no BGR conversion anywhere)."""
    _require_pillow()
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(array)
    if encoding == "rgb8":
        if arr.dtype != np.uint8 or arr.ndim != 3 or arr.shape[2] != 3:
            raise ImageCodecError(
                "RGB PNG requires uint8[H,W,3]",
                code=ErrorCode.INVALID_DTYPE,
                details={"dtype": str(arr.dtype), "shape": list(arr.shape)},
            )
    elif encoding == "depth16":
        if arr.dtype != np.uint16 or arr.ndim != 2:
            raise ImageCodecError(
                "depth PNG requires uint16[H,W]",
                code=ErrorCode.INVALID_DTYPE,
                details={"dtype": str(arr.dtype), "shape": list(arr.shape)},
            )
    else:
        raise ImageCodecError(
            f"unsupported image encoding {encoding!r}",
            code=ErrorCode.INVALID_ENCODING,
        )
    Image.fromarray(arr).save(path, format="PNG", compress_level=1)
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def read_png(path: Path, *, encoding: str) -> np.ndarray:
    _require_pillow()
    image = Image.open(path)
    arr = np.asarray(image)
    if encoding == "rgb8":
        return np.ascontiguousarray(arr, dtype=np.uint8)
    if encoding == "depth16":
        return np.ascontiguousarray(arr, dtype=np.uint16)
    raise ImageCodecError(
        f"unsupported image encoding {encoding!r}",
        code=ErrorCode.INVALID_ENCODING,
    )


__all__ = ["ImageCodecError", "read_png", "write_png"]
