"""Task 2: compare the basic image-processing operations and parameters.

The clean scene is used as the single input.  The script writes grayscale,
Gaussian blur, fixed / adaptive threshold, Canny and morphological results to
``outputs/task2`` and a 6-panel comparison image.  A CSV file records the
white-pixel / edge-pixel ratios so that the parameter discussion is based on
measurements rather than impressions.

Run from the project root:
    python src/task2_processing.py
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

from image_io import imread, imwrite

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "images" / "scene_01_clean.png"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "task2"
PANEL_W, PANEL_H = 360, 270
LABEL_H = 30


def _panel(image: np.ndarray, label: str) -> np.ndarray:
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    panel = cv2.resize(image, (PANEL_W, PANEL_H), interpolation=cv2.INTER_AREA)
    panel = cv2.copyMakeBorder(panel, LABEL_H, 0, 0, 0,
                               cv2.BORDER_CONSTANT, value=(245, 245, 245))
    cv2.putText(panel, label, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                (30, 30, 30), 1, cv2.LINE_AA)
    return panel


def _save(output_dir: Path, name: str, image: np.ndarray) -> None:
    if not imwrite(output_dir / name, image):
        raise RuntimeError(f"failed to write {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    image = imread(args.input)
    if image is None:
        raise SystemExit(f"could not read {args.input}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur5 = cv2.GaussianBlur(gray, (5, 5), 0)
    blur15 = cv2.GaussianBlur(gray, (15, 15), 0)

    # Fixed thresholds: white = pixels darker than the threshold (target).
    _, th80 = cv2.threshold(blur5, 80, 255, cv2.THRESH_BINARY_INV)
    _, th140 = cv2.threshold(blur5, 140, 255, cv2.THRESH_BINARY_INV)
    _, th200 = cv2.threshold(blur5, 200, 255, cv2.THRESH_BINARY_INV)
    adaptive = cv2.adaptiveThreshold(
        blur5, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
        blockSize=31, C=5
    )

    canny_low = cv2.Canny(blur5, 40, 120)
    canny_high = cv2.Canny(blur5, 100, 200)

    kernel5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel9 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    opened = cv2.morphologyEx(th140, cv2.MORPH_OPEN, kernel5)
    closed = cv2.morphologyEx(th140, cv2.MORPH_CLOSE, kernel9)

    # A deliberately corrupted binary mask makes the effect of morphology
    # visible: salt noise in the background + small holes inside the target.
    rng = np.random.default_rng(7)
    noisy = th140.copy()
    noise_y = rng.integers(0, noisy.shape[0], 1500)
    noise_x = rng.integers(0, noisy.shape[1], 1500)
    noisy[noise_y, noise_x] = 255
    for hx, hy in [(120, 435), (160, 465), (205, 445), (230, 485)]:
        cv2.rectangle(noisy, (hx, hy), (hx + 7, hy + 7), 0, -1)
    opened_demo = cv2.morphologyEx(noisy, cv2.MORPH_OPEN, kernel5)
    closed_demo = cv2.morphologyEx(opened_demo, cv2.MORPH_CLOSE, kernel9)

    results = {
        "01_gray.png": gray,
        "02_gaussian_5x5.png": blur5,
        "03_gaussian_15x15.png": blur15,
        "04_threshold_80.png": th80,
        "05_threshold_140.png": th140,
        "06_threshold_200.png": th200,
        "07_adaptive.png": adaptive,
        "08_canny_40_120.png": canny_low,
        "09_canny_100_200.png": canny_high,
        "10_morph_open_5x5.png": opened,
        "11_morph_close_9x9.png": closed,
        "12_morph_noisy.png": noisy,
        "13_morph_open_demo.png": opened_demo,
        "14_morph_close_demo.png": closed_demo,
    }
    for name, data in results.items():
        _save(args.output_dir, name, data)

    rows = []
    for name, data in results.items():
        values = np.unique(data)
        is_binary = values.size <= 2 and set(values.tolist()).issubset({0, 255})
        rows.append({
            "output": name,
            "shape": f"{data.shape[1]}x{data.shape[0]}",
            "mean": f"{float(data.mean()):.2f}",
            "std": f"{float(data.std()):.2f}",
            "foreground_ratio": f"{float((data > 0).mean()):.4f}" if is_binary else "n/a",
        })
    with (args.output_dir / "metrics.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f, fieldnames=["output", "shape", "mean", "std", "foreground_ratio"]
        )
        writer.writeheader()
        writer.writerows(rows)

    comparison = np.vstack([
        np.hstack([
            _panel(gray, "1. gray"),
            _panel(blur5, "2. Gaussian 5x5"),
            _panel(th140, "3. fixed threshold 140 (inv)"),
        ]),
        np.hstack([
            _panel(canny_low, "4. Canny 40/120"),
            _panel(canny_high, "5. Canny 100/200"),
            _panel(closed, "6. threshold 140 + close 9x9"),
        ]),
    ])
    _save(args.output_dir, "12_comparison.png", comparison)

    morph_demo = np.hstack([
        _panel(noisy, "a. binary + salt noise + holes"),
        _panel(opened_demo, "b. open 5x5 -> salt removed"),
        _panel(closed_demo, "c. then close 9x9 -> holes filled"),
    ])
    _save(args.output_dir, "15_morphology_demo.png", morph_demo)

    print(f"input: {args.input.name} ({image.shape[1]}x{image.shape[0]})")
    for row in rows:
        print(f"  {row['output']:<26} foreground_ratio={row['foreground_ratio']}")
    print(f"outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
