"""Task 1: basic OpenCV image operations.

For every PNG in ``data/images``:
  * read the image and verify that it was decoded,
  * print width / height / channel count,
  * resize it to 400 px width, crop the central 60 %, convert to grayscale,
  * save each result under ``outputs/task1``,
  * read the saved files back and verify shape / pixel equality.

The original images are never modified.

Run from the project root:
    python src/task1_basics.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from image_io import imread, imwrite

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "images"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "task1"
PANEL_W, PANEL_H = 320, 240
LABEL_H = 30


def _panel(image: np.ndarray, label: str) -> np.ndarray:
    panel = cv2.resize(image, (PANEL_W, PANEL_H), interpolation=cv2.INTER_AREA)
    panel = cv2.copyMakeBorder(panel, LABEL_H, 0, 0, 0,
                               cv2.BORDER_CONSTANT, value=(245, 245, 245))
    cv2.putText(panel, label, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (30, 30, 30), 1, cv2.LINE_AA)
    return panel


def process_one(path: Path, output_dir: Path) -> list[str]:
    image = imread(path)
    if image is None:
        raise RuntimeError(f"could not read {path}")
    if image.ndim != 3 or image.shape[2] != 3:
        raise RuntimeError(f"expected a 3-channel BGR image, got {image.shape}")

    h, w, channels = image.shape
    lines = [f"{path.name}: read OK, width={w}, height={h}, channels={channels}"]

    new_w = 400
    new_h = max(1, round(h * new_w / w))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Central 60 % crop, computed from the resized image.
    ch, cw = resized.shape[:2]
    crop_w, crop_h = int(cw * 0.6), int(ch * 0.6)
    x0, y0 = (cw - crop_w) // 2, (ch - crop_h) // 2
    cropped = resized[y0:y0 + crop_h, x0:x0 + crop_w].copy()

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    outputs = {
        f"resized_{path.stem}.png": resized,
        f"crop_{path.stem}.png": cropped,
        f"gray_{path.stem}.png": gray,
    }
    for name, data in outputs.items():
        if not imwrite(output_dir / name, data):
            raise RuntimeError(f"failed to write {name}")
        reloaded = imread(output_dir / name, cv2.IMREAD_UNCHANGED)
        same_shape = reloaded is not None and reloaded.shape == data.shape
        same_pixels = same_shape and np.array_equal(reloaded, data)
        lines.append(
            f"  {name}: shape={data.shape}, readback_shape_ok={same_shape}, "
            f"pixel_equal={same_pixels}"
        )
        if not (same_shape and same_pixels):
            raise RuntimeError(f"read-back check failed for {name}")

    comparison = np.hstack([
        _panel(resized, "1. resized 400px"),
        _panel(cropped, "2. center crop 60%"),
        _panel(gray_bgr, "3. grayscale"),
    ])
    if not imwrite(output_dir / f"comparison_{path.stem}.png", comparison):
        raise RuntimeError("failed to write comparison image")
    lines.append(f"  comparison written: comparison_{path.stem}.png")

    # Explicitly verify that the source file is untouched.
    original_again = imread(path)
    lines.append(f"  original unchanged: {np.array_equal(original_again, image)}")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    images = sorted(args.input_dir.glob("*.png"))
    if not images:
        raise SystemExit(f"no PNG images found in {args.input_dir}")

    report: list[str] = []
    for path in images:
        report.extend(process_one(path, args.output_dir))
    report.append(f"processed {len(images)} images; outputs in {args.output_dir}")

    text = "\n".join(report) + "\n"
    (args.output_dir / "readback_report.txt").write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
