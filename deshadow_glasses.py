# deshadow_glasses.py
# 作用: 自动去掉镜片阴影, 只保留镜框/鼻托/镜腿边缘, 导出 RGBA PNG

import argparse, os
import cv2
import numpy as np

def autocrop_rgba(rgba, pad=2):
    a = rgba[:,:,3]
    ys, xs = np.where(a > 0)
    if len(xs) == 0:
        return rgba
    y0, y1 = max(0, ys.min()-pad), min(rgba.shape[0], ys.max()+pad+1)
    x0, x1 = max(0, xs.min()-pad), min(rgba.shape[1], xs.max()+pad+1)
    return rgba[y0:y1, x0:x1]

def build_edge_alpha(img_bgr, edge_lo=60, edge_hi=160, dilate=5, feather=2,
                     band_top=0.15, band_bot=0.95):
    """
    基于白/浅底图: Canny 边缘 + 膨胀 + 羽化 -> 作为 alpha。
    用 band 截去上下无用区域, 只保留镜框所在水平带。
    """
    h, w = img_bgr.shape[:2]
    gray  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, edge_lo, edge_hi)
    if dilate > 0:
        edges = cv2.dilate(edges,
                           cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(dilate, dilate)), 1)
    band = np.zeros_like(edges)
    band[int(band_top*h):int(band_bot*h), :] = 255
    edges = cv2.bitwise_and(edges, band)
    alpha = edges
    if feather > 0:
        alpha = cv2.GaussianBlur(alpha, (0,0), feather)
    rgba = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2BGRA)
    rgba[:,:,3] = alpha
    return rgba

def deshadow_rgba_keep_rim(rgba, rim_px=9, fade=1.6, lens_alpha=0):
    """
    让 alpha 只在“边沿”处高，镜片内部渐变到低/透明。
    - rim_px: 保留边沿宽度(像素)
    - fade:   衰减指数(1.2~2.0)，越大越“只剩边”
    - lens_alpha: 镜片内部的最低不透明度(0=完全透明)
    同时把 RGB 也按新的 alpha 比例淡化，避免残留阴影颜色。
    """
    a = rgba[:,:,3].copy()
    # 若 alpha 几乎全不透明，先用边缘重建一份
    if (a > 250).mean() > 0.98:
        rgba = build_edge_alpha(rgba[:,:,:3])
        a = rgba[:,:,3]

    # 到边缘的距离(二值+距离变换)；边缘=1, 内部随距离衰减
    edge = (a > 10).astype(np.uint8)
    dist = cv2.distanceTransform(edge, cv2.DIST_L2, 5)
    rim = np.clip((rim_px - dist) / max(1e-6, rim_px), 0.0, 1.0)
    rim = rim ** float(fade)

    a_new = (np.maximum(rim, lens_alpha/255.0) * 255.0).astype(np.uint8)

    # 同步淡化 RGB (避免阴影色)
    rgb = rgba[:,:,:3].astype(np.float32)
    rgb *= (a_new[...,None] / 255.0)
    out = np.dstack([rgb.astype(np.uint8), a_new])
    return out

def main():
    ap = argparse.ArgumentParser(
        description="去掉镜片阴影/做成 RGBA (适合白底/浅底眼镜图)")
    ap.add_argument("inp", help="输入图片(jpg/png, 是否带透明通道都可)")
    ap.add_argument("-o", "--out", default=None, help="输出PNG路径(默认同名加_clean.png)")
    ap.add_argument("--rim", type=int, default=9, help="保留边沿宽度(像素)")
    ap.add_argument("--fade", type=float, default=1.6, help="边沿向内部的衰减指数(1.2~2.0)")
    ap.add_argument("--lens-alpha", type=int, default=0, help="镜片内部最低不透明度(0~255)")
    ap.add_argument("--edge-lo", type=int, default=60, help="Canny 低阈值")
    ap.add_argument("--edge-hi", type=int, default=160, help="Canny 高阈值")
    ap.add_argument("--dilate", type=int, default=5, help="边缘膨胀核大小")
    ap.add_argument("--feather", type=int, default=2, help="alpha 羽化(像素)")
    ap.add_argument("--band-top", type=float, default=0.15, help="保留水平带-上界(0~1)")
    ap.add_argument("--band-bot", type=float, default=0.95, help="保留水平带-下界(0~1)")
    ap.add_argument("--no-autocrop", action="store_true", help="不裁掉透明边")
    args = ap.parse_args()

    img = cv2.imread(args.inp, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise SystemExit(f"读取失败: {args.inp}")

    # 统一为 RGBA；若原alpha几乎全不透明，转用边缘alpha
    if img.ndim == 3 and img.shape[2] == 3:
        rgba = build_edge_alpha(img,
                                edge_lo=args.edge_lo, edge_hi=args.edge_hi,
                                dilate=args.dilate, feather=args.feather,
                                band_top=args.band_top, band_bot=args.band_bot)
    elif img.ndim == 3 and img.shape[2] == 4:
        a = img[:,:,3]
        if (a > 250).mean() > 0.98:
            rgba = build_edge_alpha(img[:,:,:3],
                                    edge_lo=args.edge_lo, edge_hi=args.edge_hi,
                                    dilate=args.dilate, feather=args.feather,
                                    band_top=args.band_top, band_bot=args.band_bot)
        else:
            rgba = img
    else:
        raise SystemExit(f"不支持的通道数: {img.shape}")

    # 去阴影(只保边沿)
    rgba = deshadow_rgba_keep_rim(rgba,
                                  rim_px=args.rim,
                                  fade=args.fade,
                                  lens_alpha=args.lens_alpha)

    if not args.no_autocrop:
        rgba = autocrop_rgba(rgba, pad=2)

    out = args.out or os.path.splitext(args.inp)[0] + "_clean.png"
    ok = cv2.imwrite(out, rgba)
    print(f"[OK] saved -> {out} ({'x'.join(map(str, rgba.shape[1::-1]))}, RGBA)")
    if not ok:
        raise SystemExit("写文件失败")

if __name__ == "__main__":
    main()
