"""Generate the four synthetic test photographs used in week 1.

The images are rendered from code so that the experiments are reproducible and
the ground-truth bounding box of the blue bottle is known.  They deliberately
cover different backgrounds, illumination levels and target sizes:

    scene_01_clean.png    bright desk, large target
    scene_02_warm.png     warm wooden table, medium target
    scene_03_lowlight.png dark room, low illumination
    scene_04_overexposed.png pale target under strong glare (failure case)

Run from the project root:
    python src/make_test_images.py
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from image_io import imwrite

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_DIR = PROJECT_ROOT / "data" / "images"
RNG = np.random.default_rng(20261022)
H, W = 600, 800


def _gradient(top: tuple[int, int, int], bottom: tuple[int, int, int]) -> np.ndarray:
    """Vertical BGR gradient with shape (H, W, 3)."""
    t = np.linspace(0.0, 1.0, H, dtype=np.float32)[:, None, None]
    top_arr = np.array(top, dtype=np.float32)[None, None, :]
    bottom_arr = np.array(bottom, dtype=np.float32)[None, None, :]
    return np.repeat(top_arr * (1.0 - t) + bottom_arr * t, W, axis=1)


def _add_radial_light(img: np.ndarray, center: tuple[int, int], radius: float,
                      colour: tuple[int, int, int], strength: float) -> np.ndarray:
    """Screen-style radial light, useful for lamps / window light."""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    d2 = (xx - center[0]) ** 2 + (yy - center[1]) ** 2
    glow = np.exp(-d2 / (2.0 * radius ** 2))[..., None] * strength
    out = img.astype(np.float32) + np.array(colour, dtype=np.float32) * glow
    return np.clip(out, 0, 255)


def _add_grain(img: np.ndarray, sigma: float = 3.0) -> np.ndarray:
    noise = RNG.normal(0.0, sigma, img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + noise, 0, 255)


def _vignette(img: np.ndarray, amount: float = 0.25) -> np.ndarray:
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    nx = (xx - W / 2) / (W / 2)
    ny = (yy - H / 2) / (H / 2)
    r2 = nx ** 2 + ny ** 2
    mask = np.clip(1.0 - amount * r2, 0.0, 1.0)[..., None]
    return img.astype(np.float32) * mask


def _bottle_layer(cx: int, cy: int, bw: int, bh: int,
                  colour: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Return (BGR layer, alpha mask) for a blue plastic bottle."""
    mask = np.zeros((H, W), dtype=np.uint8)
    body_top = cy + int(bh * 0.06)
    bottom = cy + bh // 2
    neck_top = cy - int(bh * 0.30)
    cap_top = cy - bh // 2
    neck_w = int(bw * 0.44)
    cap_w = int(bw * 0.52)

    # Rounded body.
    cv2.rectangle(mask, (cx - bw // 2, body_top), (cx + bw // 2, bottom), 255, -1)
    cv2.ellipse(mask, (cx, body_top), (bw // 2, max(8, bw // 7)),
                0, 180, 360, 255, -1)
    cv2.ellipse(mask, (cx, bottom), (bw // 2, max(8, bw // 9)),
                0, 0, 180, 255, -1)
    # Neck and shoulder.
    cv2.rectangle(mask, (cx - neck_w // 2, neck_top),
                  (cx + neck_w // 2, body_top + 2), 255, -1)
    cv2.ellipse(mask, (cx, body_top), (neck_w // 2, max(6, bw // 9)),
                0, 180, 360, 255, -1)
    # Cap.
    cv2.rectangle(mask, (cx - cap_w // 2, cap_top),
                  (cx + cap_w // 2, neck_top), 255, -1)
    cv2.ellipse(mask, (cx, cap_top), (cap_w // 2, max(6, bw // 12)),
                0, 0, 180, 255, -1)

    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    u = (xx - cx) / max(bw, 1)
    v = (yy - cy) / max(bh, 1)

    # Cylindrical shading: light from the left, darker rim on the right.
    shade = 0.62 + 0.46 * np.exp(-((u + 0.28) / 0.48) ** 2)
    shade *= 0.92 + 0.12 * np.clip(-v, -0.5, 0.8)
    base = np.array(colour, dtype=np.float32)[None, None, :]
    layer = base * shade[..., None]

    # Specular highlight and a faint warm reflection near the bottom.
    spec = np.exp(-((u + 0.30) / 0.075) ** 2) * np.exp(-((v + 0.08) / 0.60) ** 2)
    layer += spec[..., None] * np.array([120, 120, 120], dtype=np.float32)
    layer += np.clip(-v, 0, 1)[..., None] * np.array([8, 14, 20], dtype=np.float32)

    # Slightly different cap colour.
    cap_zone = (yy <= neck_top + 2).astype(np.float32)[..., None]
    layer = layer * (1.0 - 0.30 * cap_zone) + np.array([18, 20, 28], dtype=np.float32) * 0.30 * cap_zone

    layer = np.clip(layer, 0, 255)
    alpha = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (5, 5), 1.1)
    return layer, alpha


def _composite_bottle(img: np.ndarray, cx: int, cy: int, bw: int, bh: int,
                      colour: tuple[int, int, int], shadow_alpha: float = 0.45,
                      shadow_offset: tuple[int, int] = (16, 10)) -> tuple[np.ndarray, list[int]]:
    layer, alpha = _bottle_layer(cx, cy, bw, bh, colour)

    # Soft cast shadow on the table.
    silhouette = (alpha * 255).astype(np.uint8)
    shadow = np.zeros_like(silhouette)
    sh_x, sh_y = shadow_offset
    shadow[max(0, sh_y):H, max(0, sh_x):W] = silhouette[0:H - sh_y, 0:W - sh_x]
    shadow = cv2.GaussianBlur(shadow.astype(np.float32) / 255.0, (0, 0), 16)
    img = img.astype(np.float32) * (1.0 - shadow_alpha * shadow[..., None])

    out = img * (1.0 - alpha[..., None]) + layer * alpha[..., None]
    # Use the rendered alpha to obtain the true visible extent.  The bottle
    # body has a rounded bottom, so cy +/- bh/2 would underestimate its bbox.
    ys, xs = np.where(alpha > 0.5)
    if xs.size:
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        bbox = [x0, y0, x1 - x0 + 1, y1 - y0 + 1]
    else:
        bbox = [int(cx - bw / 2), int(cy - bh / 2), int(bw), int(bh)]
    return np.clip(out, 0, 255), bbox


def scene_clean() -> tuple[np.ndarray, list[int]]:
    img = _gradient((232, 235, 237), (196, 200, 204))
    # Subtle desk surface texture.
    for y in range(0, H, 46):
        cv2.line(img, (0, y), (W, y), (188, 192, 196), 1)
    img = _add_grain(img, 2.4)

    # Notebook, pen and a mug in the background.
    cv2.rectangle(img, (72, 392), (300, 530), (126, 132, 138), -1)
    cv2.rectangle(img, (72, 392), (300, 530), (86, 92, 98), 3)
    for i in range(4):
        cv2.line(img, (92, 420 + i * 24), (278, 420 + i * 24), (105, 112, 118), 2)
    cv2.line(img, (330, 500), (470, 540), (26, 50, 110), 9, cv2.LINE_AA)
    cv2.circle(img, (330, 500), 5, (60, 80, 140), -1, cv2.LINE_AA)
    cv2.ellipse(img, (680, 410), (55, 70), 0, 180, 360, (150, 155, 160), -1)
    cv2.rectangle(img, (625, 410), (735, 500), (150, 155, 160), -1)

    img = _add_radial_light(img, (90, 60), 260, (28, 26, 22), 0.55)
    img, bbox = _composite_bottle(img, 390, 305, 118, 236, (190, 98, 36))
    img = _vignette(img, 0.14)
    return np.clip(img, 0, 255).astype(np.uint8), bbox


def scene_warm() -> tuple[np.ndarray, list[int]]:
    img = _gradient((62, 112, 168), (38, 78, 126))
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    # Wood grain: wavy horizontal bands plus a few knots.
    grain = 10 * np.sin(xx / 17.0 + 3.0 * np.sin(yy / 31.0)) + 7 * np.sin(xx / 6.0)
    img += grain[..., None] * np.array([0.45, 0.65, 1.0], dtype=np.float32)
    for y in range(40, H, 95):
        cv2.line(img, (0, y), (W, y + 3), (30, 65, 110), 2)
    for kx, ky in [(120, 480), (700, 120), (560, 540)]:
        cv2.ellipse(img, (kx, ky), (32, 10), 12, 0, 360, (34, 70, 112), -1)
        cv2.ellipse(img, (kx, ky), (18, 6), 12, 0, 360, (46, 88, 132), -1)
    img = _add_grain(img, 3.4)

    # Red book and a grey pen: colourful clutter, but no blue distractor.
    cv2.rectangle(img, (80, 400), (335, 520), (48, 52, 128), -1)
    cv2.rectangle(img, (80, 400), (335, 520), (28, 30, 86), 3)
    cv2.line(img, (112, 442), (300, 442), (70, 76, 150), 2)
    cv2.line(img, (112, 470), (260, 470), (70, 76, 150), 2)
    cv2.line(img, (300, 540), (470, 585), (80, 80, 80), 8, cv2.LINE_AA)

    img = _add_radial_light(img, (690, 80), 300, (30, 80, 120), 0.55)
    img, bbox = _composite_bottle(img, 535, 330, 92, 186, (165, 85, 30),
                                  shadow_alpha=0.50, shadow_offset=(20, 12))
    img = _vignette(img, 0.26)
    return np.clip(img, 0, 255).astype(np.uint8), bbox


def scene_lowlight() -> tuple[np.ndarray, list[int]]:
    img = _gradient((28, 26, 24), (12, 12, 14))
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    img += (6 * np.sin(xx / 55.0) + 4 * np.cos(yy / 37.0))[..., None]
    # Window / lamp glow at upper-left.
    img = _add_radial_light(img, (160, 130), 250, (95, 105, 115), 0.95)
    # Table edge.
    cv2.line(img, (0, 430), (W, 470), (55, 50, 45), 4, cv2.LINE_AA)
    # Monitor edge on the right, with a small blue LED (a deliberate distractor).
    cv2.rectangle(img, (650, 160), (795, 400), (24, 24, 26), -1)
    cv2.rectangle(img, (650, 160), (795, 400), (40, 40, 44), 3)
    cv2.circle(img, (690, 380), 4, (230, 120, 40), -1, cv2.LINE_AA)

    img, bbox = _composite_bottle(img, 320, 330, 126, 232, (178, 92, 34),
                                  shadow_alpha=0.55, shadow_offset=(18, 12))
    img = np.clip(img * 0.52, 0, 255)
    img = _add_radial_light(img, (180, 180), 300, (40, 42, 36), 0.55)
    img = _add_grain(img, 5.0)
    img = _vignette(img, 0.32)
    return np.clip(img, 0, 255).astype(np.uint8), bbox


def scene_overexposed() -> tuple[np.ndarray, list[int]]:
    """Pale blue bottle in a bright, glare-filled scene.

    The bottle is clearly visible to a human, but its saturation is below the
    S>=90 threshold used by the detector.  This is the documented false
    negative (missed detection) case.
    """
    img = _gradient((252, 250, 245), (224, 221, 214))
    for y in range(0, H, 50):
        cv2.line(img, (0, y), (W, y), (214, 211, 205), 1)
    # Very bright window on the right.
    img = _add_radial_light(img, (740, 60), 320, (60, 58, 52), 1.0)
    # A neutral grey notebook, far from the target hue.
    cv2.rectangle(img, (75, 405), (300, 525), (178, 176, 172), -1)
    cv2.rectangle(img, (75, 405), (300, 525), (146, 144, 140), 3)
    cv2.line(img, (100, 445), (270, 445), (158, 156, 152), 2)
    cv2.line(img, (100, 475), (240, 475), (158, 156, 152), 2)

    img, bbox = _composite_bottle(img, 430, 325, 104, 200, (215, 178, 154),
                                  shadow_alpha=0.30, shadow_offset=(18, 10))
    # Extra overall exposure pushes the target further towards white.
    img = np.clip(img * 1.06 + 6, 0, 255)
    img = _add_grain(img, 2.2)
    img = _vignette(img, 0.08)
    return np.clip(img, 0, 255).astype(np.uint8), bbox


def main() -> None:
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    scenes = [
        ("scene_01_clean.png", scene_clean, "bright desk, large target"),
        ("scene_02_warm.png", scene_warm, "warm wooden table, medium target"),
        ("scene_03_lowlight.png", scene_lowlight, "dark room, low illumination"),
        ("scene_04_overexposed.png", scene_overexposed, "pale target under strong glare"),
    ]
    ground_truth = {}
    for name, func, note in scenes:
        image, bbox = func()
        path = IMAGE_DIR / name
        if not imwrite(path, image):
            raise RuntimeError(f"failed to write {path}")
        x, y, w, h = bbox
        ground_truth[name] = {
            "note": note,
            "bbox_xywh": bbox,
            "center_xy": [x + w // 2, y + h // 2],
        }
        print(f"wrote {path.name}: {image.shape[1]}x{image.shape[0]}, bbox={bbox}")
    with (IMAGE_DIR.parent / "ground_truth.json").open("w", encoding="utf-8") as f:
        json.dump(ground_truth, f, ensure_ascii=False, indent=2)
    print(f"wrote {IMAGE_DIR.parent / 'ground_truth.json'}")


if __name__ == "__main__":
    main()
