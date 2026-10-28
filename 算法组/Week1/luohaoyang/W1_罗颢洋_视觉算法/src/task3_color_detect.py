"""Task 3: blue-object extraction and contour localisation.

Pipeline for every test image:
    BGR -> Gaussian blur -> HSV -> inRange -> open -> close
        -> findContours -> area filter -> largest contour
        -> bounding rectangle + image-moment centre -> annotate + save

Outputs (under ``outputs/task3``):
    mask_<scene>.png, result_<scene>.png for every scene,
    mask.png / result.png as copies of the clean representative result,
    detections.csv with centre, area, bbox, contour counts and IoU,
    overview.png contact sheet.

Run from the project root:
    python src/task3_color_detect.py
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import cv2
import numpy as np

from image_io import imread, imwrite

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "images"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "task3"
GROUND_TRUTH = PROJECT_ROOT / "data" / "ground_truth.json"

# ---- Parameters (explained in README) -------------------------------------
HSV_LOWER = np.array([95, 90, 35], dtype=np.uint8)   # H, S, V
HSV_UPPER = np.array([135, 255, 255], dtype=np.uint8)
MIN_AREA = 600          # px^2, at 800x600 resolution
OPEN_KERNEL = (5, 5)    # remove salt noise / thin reflections
CLOSE_KERNEL = (7, 7)   # fill small holes inside the object
BLUR_KERNEL = (3, 3)
IOU_OK = 0.50

PANEL_W, PANEL_H = 400, 300
LABEL_H = 30


def _iou(box_a: list[int], box_b: list[int]) -> float:
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = aw * ah + bw * bh - inter
    return float(inter / union) if union else 0.0


def _annotate(image: np.ndarray, contour: np.ndarray, bbox: list[int],
              centre: tuple[int, int], area: float) -> np.ndarray:
    out = image.copy()
    x, y, w, h = bbox
    cx, cy = centre

    cv2.drawContours(out, [contour], -1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2, cv2.LINE_AA)
    cv2.drawMarker(out, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 22, 2, cv2.LINE_AA)
    cv2.circle(out, (cx, cy), 4, (0, 0, 255), -1, cv2.LINE_AA)

    label = f"center=({cx},{cy}) area={int(area)}"
    label_y = y - 10 if y > 30 else y + h + 24
    cv2.putText(out, label, (x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(out, "blue target", (x, max(22, label_y - 24)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
    return out


def _panel(image: np.ndarray, label: str) -> np.ndarray:
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    panel = cv2.resize(image, (PANEL_W, PANEL_H), interpolation=cv2.INTER_AREA)
    panel = cv2.copyMakeBorder(panel, LABEL_H, 0, 0, 0,
                               cv2.BORDER_CONSTANT, value=(245, 245, 245))
    cv2.putText(panel, label, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                (30, 30, 30), 1, cv2.LINE_AA)
    return panel


def process_one(path: Path, output_dir: Path,
                ground_truth: dict) -> tuple[dict, np.ndarray, np.ndarray]:
    image = imread(path)
    if image is None:
        raise RuntimeError(f"could not read {path}")

    blurred = cv2.GaussianBlur(image, BLUR_KERNEL, 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, HSV_LOWER, HSV_UPPER)

    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, OPEN_KERNEL)
    )
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, CLOSE_KERNEL)
    )

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    kept = [c for c in contours if cv2.contourArea(c) >= MIN_AREA]

    row = {
        "image": path.name,
        "status": "MISS",
        "center_x": "",
        "center_y": "",
        "area_px": "",
        "bbox_x": "",
        "bbox_y": "",
        "bbox_w": "",
        "bbox_h": "",
        "contours_total": len(contours),
        "contours_kept": len(kept),
        "iou": "0.0000",
    }

    if not kept:
        result = image.copy()
        cv2.putText(result, "NO TARGET: HSV mask area < 600 px", (24, 46),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.78, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(result, "Check saturation / exposure (see README)", (24, 82),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 255), 1, cv2.LINE_AA)
        if not imwrite(output_dir / f"mask_{path.stem}.png", mask):
            raise RuntimeError("failed to write mask")
        if not imwrite(output_dir / f"result_{path.stem}.png", result):
            raise RuntimeError("failed to write result")
        return row, mask, result

    contour = kept[0]
    area = float(cv2.contourArea(contour))
    x, y, w, h = cv2.boundingRect(contour)
    moments = cv2.moments(contour)
    if moments["m00"] > 0:
        cx = int(round(moments["m10"] / moments["m00"]))
        cy = int(round(moments["m01"] / moments["m00"]))
    else:
        cx, cy = x + w // 2, y + h // 2

    result = _annotate(image, contour, [x, y, w, h], (cx, cy), area)

    gt = ground_truth.get(path.name, {})
    iou = _iou([x, y, w, h], gt["bbox_xywh"]) if gt else 0.0
    row.update({
        "status": "OK" if iou >= IOU_OK else "LOW_IOU",
        "center_x": cx,
        "center_y": cy,
        "area_px": int(area),
        "bbox_x": x,
        "bbox_y": y,
        "bbox_w": w,
        "bbox_h": h,
        "iou": f"{iou:.4f}",
    })

    if not imwrite(output_dir / f"mask_{path.stem}.png", mask):
        raise RuntimeError("failed to write mask")
    if not imwrite(output_dir / f"result_{path.stem}.png", result):
        raise RuntimeError("failed to write result")
    return row, mask, result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    images = sorted(args.input_dir.glob("*.png"))
    if not images:
        raise SystemExit(f"no PNG images found in {args.input_dir}")

    ground_truth = {}
    if GROUND_TRUTH.exists():
        ground_truth = json.loads(GROUND_TRUTH.read_text(encoding="utf-8"))

    rows, masks, results = [], [], []
    for path in images:
        row, mask, result = process_one(path, args.output_dir, ground_truth)
        rows.append(row)
        masks.append((path.stem, mask))
        results.append((path.stem, result))
        print(f"{path.name:<32} status={row['status']:<7} "
              f"center=({row['center_x']},{row['center_y']}) "
              f"area={row['area_px']} bbox=({row['bbox_x']},{row['bbox_y']},"
              f"{row['bbox_w']},{row['bbox_h']}) IoU={row['iou']}")

    fieldnames = list(rows[0].keys())
    with (args.output_dir / "detections.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    params = {
        "hsv_lower": HSV_LOWER.tolist(),
        "hsv_upper": HSV_UPPER.tolist(),
        "min_area_px": MIN_AREA,
        "open_kernel": list(OPEN_KERNEL),
        "close_kernel": list(CLOSE_KERNEL),
        "blur_kernel": list(BLUR_KERNEL),
        "iou_ok": IOU_OK,
    }
    (args.output_dir / "params.json").write_text(
        json.dumps(params, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Contact sheet: mask / result for every scene.
    labels = {
        "scene_01_clean": "clean / bright",
        "scene_02_warm": "warm wood",
        "scene_03_lowlight": "low light",
        "scene_04_overexposed": "overexposed (failure)",
    }
    sheet_rows = []
    for (stem, mask), (_, result) in zip(masks, results):
        sheet_rows.append(np.hstack([
            _panel(mask, f"{labels.get(stem, stem)} - mask"),
            _panel(result, f"{labels.get(stem, stem)} - result"),
        ]))
    overview = np.vstack(sheet_rows)
    if not imwrite(args.output_dir / "overview.png", overview):
        raise RuntimeError("failed to write overview.png")

    # Representative single files required by the assignment.
    clean_stem = "scene_01_clean"
    if any(stem == clean_stem for stem, _ in masks):
        shutil.copy2(args.output_dir / f"mask_{clean_stem}.png", args.output_dir / "mask.png")
        shutil.copy2(args.output_dir / f"result_{clean_stem}.png", args.output_dir / "result.png")

    ok = sum(1 for r in rows if r["status"] == "OK")
    print(f"\n{ok}/{len(rows)} scenes passed IoU >= {IOU_OK}; outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
