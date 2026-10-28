"""学习内容配套代码：从读图到轮廓定位的一条完整 OpenCV 流水线。

这段代码对应任务书“2. 学习内容”表格中的 6 个板块：
    基础操作 -> 颜色空间 -> 图像预处理 -> 形态学处理
    -> 轮廓与几何信息 -> 结果验证

运行方式（在项目根目录）：
    python src/learning_content_demo.py

所有中间结果写到 outputs/learning/，建议一边运行一边对照注释观察变量。
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from image_io import imread, imwrite

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_PATH = PROJECT_ROOT / "data" / "images" / "scene_01_clean.png"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "learning"

# HSV 蓝色范围：H 是色调，S 是饱和度，V 是明度。
# OpenCV 中 H 的范围是 0~179（不是 0~360），蓝色大约在 100~130。
BLUE_LOWER = np.array([95, 90, 35], dtype=np.uint8)
BLUE_UPPER = np.array([135, 255, 255], dtype=np.uint8)

# 小于这个面积的轮廓当作噪声（800x600 图像下目标面积约 2 万像素）。
MIN_AREA = 600


def panel(image: np.ndarray, label: str, size: tuple[int, int] = (360, 270)) -> np.ndarray:
    """把任意单通道/三通道图缩放并加标题，方便拼成对比图。"""
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    image = cv2.copyMakeBorder(
        image, 30, 0, 0, 0, cv2.BORDER_CONSTANT, value=(245, 245, 245)
    )
    cv2.putText(image, label, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                (30, 30, 30), 1, cv2.LINE_AA)
    return image


def save(name: str, image: np.ndarray) -> None:
    """统一保存，并顺手检查是否写成功。"""
    if not imwrite(OUTPUT_DIR / name, image):
        raise RuntimeError(f"保存失败：{name}")
    print(f"  已保存 outputs/learning/{name}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ==================================================================
    # 1. 基础操作：读取、尺寸、通道数
    # ==================================================================
    # imread 返回 BGR 三通道数组，shape 的顺序是 (高, 宽, 通道数)。
    bgr = imread(IMAGE_PATH)
    if bgr is None:
        raise SystemExit(f"读不到图片：{IMAGE_PATH}")

    height, width, channels = bgr.shape
    print("=" * 60)
    print("1. 基础操作")
    print(f"  图片尺寸：width={width}, height={height}, channels={channels}")
    print(f"  dtype={bgr.dtype}, 像素最小值={bgr.min()}, 最大值={bgr.max()}")
    print(f"  总像素数={bgr.size}（= 宽 x 高 x 通道数）")

    # 取一个像素观察 BGR 三个分量。坐标写法是 img[y, x]，先 y 后 x。
    # 这里取蓝色瓶身内部的一点。
    sample_y, sample_x = 330, 390
    b, g, r = bgr[sample_y, sample_x]
    print(f"  像素 (x={sample_x}, y={sample_y}) 的 BGR = ({b}, {g}, {r})")
    print("  注意：OpenCV 的通道顺序是 BGR，不是 RGB。")

    # ==================================================================
    # 2. 颜色空间：BGR 与 RGB 的区别
    # ==================================================================
    # cvtColor 只是“换一种通道顺序/表示方法”，不会改变图像内容。
    # 把 BGR 转成 RGB 后，第 0 通道从 B 变成 R。
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    print("\n2. 颜色空间")
    print(f"  同一个像素的 RGB = {tuple(int(v) for v in rgb[sample_y, sample_x])}")

    # 故意把 RGB 数组直接交给 imwrite（imwrite 默认按 BGR 解释），
    # 这样保存出来的图会红蓝互换，用来说明“通道顺序不能混”。
    bgr_panel = panel(bgr, "correct: BGR -> imwrite")
    wrong_panel = panel(rgb, "wrong: RGB -> imwrite (R/B swapped)")
    save("bgr_vs_rgb.png", np.hstack([bgr_panel, wrong_panel]))

    # ==================================================================
    # 3. 基础操作：缩放与裁剪
    # ==================================================================
    # resize 的 dsize 是 (宽, 高)，与 shape 的 (高, 宽) 顺序相反。
    resized = cv2.resize(bgr, (400, 300), interpolation=cv2.INTER_AREA)
    # 裁剪用切片 img[y1:y2, x1:x2]，先写行（y），再写列（x）。
    cropped = bgr[150:450, 220:580].copy()
    print("\n3. 缩放与裁剪")
    print(f"  resize 后 shape={resized.shape}")
    print(f"  crop 后 shape={cropped.shape}")
    save("01_resized.png", resized)
    save("02_crop.png", cropped)

    # ==================================================================
    # 4. 颜色空间：灰度图与 HSV
    # ==================================================================
    # 灰度图只有亮度信息：Gray = 0.114B + 0.587G + 0.299R。
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    # HSV 把颜色拆成：H 色调、S 饱和度、V 明度。
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)  # 分别取出 H、S、V 三个通道

    print("\n4. 灰度与 HSV")
    print(f"  gray shape={gray.shape}, dtype={gray.dtype}")
    for name, channel in [("H", h), ("S", s), ("V", v)]:
        print(f"  {name}: min={channel.min()}, max={channel.max()}, "
              f"mean={channel.mean():.1f}")
    save("03_gray.png", gray)
    save("04_hue.png", h)
    save("05_saturation.png", s)
    save("06_value.png", v)

    # ==================================================================
    # 5. 颜色空间：inRange 生成蓝色掩膜
    # ==================================================================
    # inRange 会对每个像素做判断：三个通道都在 [lower, upper] 内才是白色。
    # 白色（255）表示“属于目标”，黑色（0）表示“背景”。
    mask = cv2.inRange(hsv, BLUE_LOWER, BLUE_UPPER)
    # bitwise_and 用掩膜保留原图中目标区域的像素，其余变黑。
    masked = cv2.bitwise_and(bgr, bgr, mask=mask)

    blue_ratio = float((mask > 0).mean())
    print("\n5. 蓝色掩膜")
    print(f"  HSV 下限={BLUE_LOWER.tolist()}, 上限={BLUE_UPPER.tolist()}")
    print(f"  掩膜白色像素占比={blue_ratio:.2%}")
    save("07_blue_mask.png", mask)
    save("08_masked.png", masked)

    # ==================================================================
    # 6. 图像预处理：均值 / 高斯 / 中值滤波
    # ==================================================================
    # 三种滤波都是“用邻域像素平滑当前像素”，区别在权重和适用噪声。
    mean_blur = cv2.blur(bgr, (5, 5))                 # 邻域等权平均
    gaussian_blur = cv2.GaussianBlur(bgr, (5, 5), 0)  # 高斯权重，边缘更自然
    median_blur = cv2.medianBlur(bgr, 5)              # 取中位数，擅长椒盐噪声

    print("\n6. 图像预处理：滤波")
    print("  均值滤波：简单、快，但边缘会变模糊")
    print("  高斯滤波：中心权重大，常用于去高斯噪声/平滑")
    print("  中值滤波：对椒盐噪声特别有效，能较好保留边缘")
    save("09_mean_blur.png", mean_blur)
    save("10_gaussian_blur.png", gaussian_blur)
    save("11_median_blur.png", median_blur)

    # ==================================================================
    # 7. 图像预处理：固定阈值、自适应阈值、Canny
    # ==================================================================
    # THRESH_BINARY_INV：比阈值暗的像素变白（目标），比阈值亮的变黑。
    _, fixed_threshold = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY_INV)
    # 自适应阈值：每个像素用邻域均值/高斯加权值做阈值，适合光照不均。
    adaptive_threshold = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
        blockSize=31, C=5
    )
    # Canny：先用低阈值找弱边缘，再用高阈值保留强边缘。
    # 低阈值越小，保留下来的边缘越多，也越容易把噪声当边缘。
    canny_low = cv2.Canny(gray, 40, 120)
    canny_high = cv2.Canny(gray, 100, 200)

    print("\n7. 阈值与边缘")
    print(f"  固定阈值 140 的前景占比={float((fixed_threshold > 0).mean()):.2%}")
    print(f"  自适应阈值的前景占比={float((adaptive_threshold > 0).mean()):.2%}")
    print(f"  Canny 40/120 边缘像素占比={float((canny_low > 0).mean()):.2%}")
    print(f"  Canny 100/200 边缘像素占比={float((canny_high > 0).mean()):.2%}")
    save("12_threshold_140.png", fixed_threshold)
    save("13_adaptive_threshold.png", adaptive_threshold)
    save("14_canny_40_120.png", canny_low)
    save("15_canny_100_200.png", canny_high)

    # ==================================================================
    # 8. 形态学处理：腐蚀、膨胀、开运算、闭运算
    # ==================================================================
    # 结构元素（kernel）决定“腐蚀/膨胀”的形状和大小，这里用椭圆。
    kernel5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    eroded = cv2.erode(mask, kernel5)       # 腐蚀：白色区域变瘦，去掉小噪点
    dilated = cv2.dilate(mask, kernel5)     # 膨胀：白色区域变胖，连接断裂
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel5)    # 先腐蚀后膨胀
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel7)  # 先膨胀后腐蚀

    print("\n8. 形态学处理")
    print(f"  原掩膜白色像素={int((mask > 0).sum())}")
    print(f"  腐蚀后白色像素={int((eroded > 0).sum())}")
    print(f"  膨胀后白色像素={int((dilated > 0).sum())}")
    print(f"  开运算后白色像素={int((opened > 0).sum())}")
    print(f"  闭运算后白色像素={int((closed > 0).sum())}")
    save("16_erode.png", eroded)
    save("17_dilate.png", dilated)
    save("18_open.png", opened)
    save("19_close.png", closed)

    # ==================================================================
    # 9. 轮廓与几何信息：findContours / contourArea / boundingRect / moments
    # ==================================================================
    # RETR_EXTERNAL 只取最外层轮廓；CHAIN_APPROX_SIMPLE 压缩直线段上的点。
    contours, _ = cv2.findContours(
        closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    # 按面积从大到小排序，最大的通常是真正目标。
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    # 面积过滤：小于 MIN_AREA 的轮廓丢弃（反光点、指示灯、噪声）。
    valid_contours = [c for c in contours if cv2.contourArea(c) >= MIN_AREA]

    print("\n9. 轮廓与几何信息")
    print(f"  找到轮廓数={len(contours)}, 面积过滤后保留={len(valid_contours)}")
    if not valid_contours:
        raise SystemExit("没有找到面积足够大的蓝色目标，请检查 HSV 范围。")

    target = valid_contours[0]
    area = cv2.contourArea(target)
    x, y, w, h = cv2.boundingRect(target)  # 外接矩形：左上角 + 宽高

    # 矩（moments）可以用来求质心：cx=m10/m00, cy=m01/m00。
    moments = cv2.moments(target)
    if moments["m00"] != 0:
        cx = int(round(moments["m10"] / moments["m00"]))
        cy = int(round(moments["m01"] / moments["m00"]))
    else:
        cx, cy = x + w // 2, y + h // 2  # 兜底：面积太小无法求质心

    print(f"  最大轮廓：面积={area:.0f} px^2")
    print(f"  外接矩形：x={x}, y={y}, w={w}, h={h}")
    print(f"  质心坐标：({cx}, {cy})")

    # 在原图上画轮廓、矩形、中心点和文字，做可视化验证。
    result = bgr.copy()
    cv2.drawContours(result, [target], -1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2, cv2.LINE_AA)
    cv2.circle(result, (cx, cy), 5, (0, 0, 255), -1, cv2.LINE_AA)
    cv2.drawMarker(result, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 22, 2,
                   cv2.LINE_AA)
    cv2.putText(result, f"area={int(area)} center=({cx},{cy})", (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)
    save("20_result.png", result)

    # ==================================================================
    # 10. 结果验证：拼对比图、回读保存结果、记录参数
    # ==================================================================
    overview = np.vstack([
        np.hstack([
            panel(bgr, "1. original BGR"),
            panel(gray, "2. gray"),
            panel(mask, "3. blue mask (inRange)"),
        ]),
        np.hstack([
            panel(gaussian_blur, "4. Gaussian 5x5"),
            panel(canny_low, "5. Canny 40/120"),
            panel(result, "6. contour + bbox + center"),
        ]),
    ])
    save("learning_overview.png", overview)

    # 回读验证：确认保存的文件能再次解码，且 shape 与内存中的结果一致。
    reloaded = imread(OUTPUT_DIR / "20_result.png")
    same_shape = reloaded is not None and reloaded.shape == result.shape
    same_pixels = same_shape and np.array_equal(reloaded, result)
    print("\n10. 结果验证")
    print(f"  回读 20_result.png: shape_ok={same_shape}, pixel_equal={same_pixels}")

    # 把关键参数和测量值写入文本，方便复现和对比不同图片。
    report = [
        f"image={IMAGE_PATH.name}",
        f"shape={bgr.shape}",
        f"BGR_pixel_at_({sample_x},{sample_y})={bgr[sample_y, sample_x].tolist()}",
        f"RGB_pixel_at_({sample_x},{sample_y})={rgb[sample_y, sample_x].tolist()}",
        f"blue_hsv_lower={BLUE_LOWER.tolist()}",
        f"blue_hsv_upper={BLUE_UPPER.tolist()}",
        f"mask_white_ratio={blue_ratio:.4f}",
        f"fixed_threshold=140",
        f"canny_low=40,120",
        f"canny_high=100,200",
        f"min_area={MIN_AREA}",
        f"target_area={area:.0f}",
        f"target_bbox={x},{y},{w},{h}",
        f"target_center={cx},{cy}",
        f"readback_shape_ok={same_shape}",
        f"readback_pixel_equal={same_pixels}",
    ]
    (OUTPUT_DIR / "learning_notes.txt").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    print(f"  已保存 outputs/learning/learning_notes.txt")
    print("=" * 60)
    print("学习内容演示结束，建议打开 outputs/learning/learning_overview.png 对照查看。")


if __name__ == "__main__":
    main()
