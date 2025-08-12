# make_transparent_glasses.py
import cv2, numpy as np, argparse, os

def to_rgba_edge(img_bgr, bg="white", edge_lo=60, edge_hi=160,
                 band_top=0.20, band_bot=0.90, dilate=5, feather=2):
    h, w = img_bgr.shape[:2]

    # 1) 近白/灰背景初筛（不一定全对，但能先去大片底色）
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    H,S,V = cv2.split(hsv)
    if bg == "white":
        bg_mask = ((V > 215) & (S < 40)).astype(np.uint8)*255
    elif bg == "black":
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        bg_mask = (gray < 30).astype(np.uint8)*255
    else:
        bg_mask = np.zeros((h,w), np.uint8)

    # 2) 边缘（保住镜框线条）
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, edge_lo, edge_hi)
    if dilate > 0:
        edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(dilate,dilate)), 1)

    # 只取中间竖向带（避免上/下无关边缘）
    band = np.zeros_like(edges)
    y0, y1 = int(band_top*h), int(band_bot*h)
    band[y0:y1, :] = 255
    edges = cv2.bitwise_and(edges, band)

    # 3) 前景 = 反背景 或 边缘（保持线条）
    fg = cv2.bitwise_or(255 - bg_mask, edges)

    # 清一下小噪点
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN,  cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)), 1)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)), 1)

    # 4) 生成 alpha 并稍微羽化
    alpha = fg
    if feather > 0:
        alpha = cv2.GaussianBlur(alpha, (0,0), feather)

    # 5) 合成 RGBA
    rgba = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2BGRA)
    rgba[:,:,3] = alpha
    return rgba

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="输入眼镜照片（png/jpg）")
    ap.add_argument("--out", default=None, help="输出文件名（默认 *_rgba.png）")
    ap.add_argument("--bg",  default="white", choices=["white","black","none"], help="背景类型（默认white）")
    ap.add_argument("--edge", nargs=2, type=int, default=[60,160], help="Canny阈值 lo hi")
    ap.add_argument("--dilate", type=int, default=5, help="边缘膨胀核（线条粗细）")
    ap.add_argument("--feather", type=float, default=2.0, help="alpha羽化sigma")
    ap.add_argument("--band", nargs=2, type=float, default=[0.20,0.90], help="边缘有效带 y_top y_bot(0~1)")
    args = ap.parse_args()

    img = cv2.imread(args.input, cv2.IMREAD_COLOR)
    assert img is not None, f"无法读取: {args.input}"

    rgba = to_rgba_edge(
        img, bg=args.bg,
        edge_lo=args.edge[0], edge_hi=args.edge[1],
        band_top=args.band[0], band_bot=args.band[1],
        dilate=args.dilate, feather=args.feather
    )

    out = args.out or os.path.splitext(args.input)[0] + "_rgba.png"
    cv2.imwrite(out, rgba)
    print("✅ 已保存：", out, rgba.shape)

if __name__ == "__main__":
    main()
