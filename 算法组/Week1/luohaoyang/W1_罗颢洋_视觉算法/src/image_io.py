"""Small wrappers around OpenCV file I/O that support non-ASCII paths on Windows.

``cv2.imread`` / ``cv2.imwrite`` use the ANSI code page on Windows and fail for
paths such as ``W1_罗颢洋_视觉算法``.  Reading with ``numpy.fromfile`` and
encoding with ``cv2.imencode`` avoids that limitation.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def imread(path: str | Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray | None:
    path = Path(path)
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


def imwrite(path: str | Path, image: np.ndarray) -> bool:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower() or ".png"
    ok, encoded = cv2.imencode(suffix, image)
    if not ok:
        return False
    encoded.tofile(str(path))
    return True
