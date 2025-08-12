# -*- coding: utf-8 -*-
# rectify_and_cut.py
# 交互取点(8点) -> 两片镜面透视矫正到正视 -> 仅保留镜框 -> 导出 RGBA(1600px)

import sys, pathlib
import numpy as np
import cv2

# ===== 可调参数 =====
W_OUT, H_OUT   = 1600, 720   # 输出画布尺寸（宽固定1600，H可按需要改）
GAP_RATIO      = 0.10        # 两镜片中心间隙占宽度比例（0.08～0.14）
LENS_W_RATIO   = 0.46        # 单侧镜片宽占 (W_OUT - gap) 比例（0.42～0.50）
LENS_H_RATIO   = 0.78        # 镜片高占 H_OUT 比例（0.70～0.85）

RIM_THICK      = 6           # 轮廓厚度估计（线条太细就加大到8~10）
KEEP_TEMPLES   = False       # 是否保留两侧镜腿（True/False）
TEMPLE_TRIM    = 0.06        # 保留镜腿时，左右边裁掉的比例（越大越短）

OUTLINE_ONLY   = True        # True=只保留边线(rim)，False=保留框体深色区域
OUT_NAME       = "Frame_A.png"


def pick_8_points(img):
    """交互式采集 8 点：左片 外上→外下→内下→内上；右片同顺序"""
    pts = []
    WIN = "pick 8 points"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, max(640, img.shape[1]//2), max(360, img.shape[0]//2))

    def on_mouse(event, x, y, flags, param):
        nonlocal pts
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 8:
            pts.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN and pts:
            pts.pop()

    cv2.setMouseCallback(WIN, on_mouse)
    print("依次点击：左片 外上→外下→内下→内上；右片同顺序")
    print("快捷键：Enter=确认(需8点)  U=撤销  R=清空  Esc=退出")

    while True:
        vis = img.copy()
        for i, (x, y) in enumerate(pts):
            cv2.circle(vis, (x, y), 5, (0, 255, 0), -1)
            cv2.putText(vis, str(i+1), (x+6, y-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2, cv2.LINE_AA)
        cv2.imshow(WIN, vis)
        k = cv2.waitKey(30) & 0xFF
        if k == 13 and len(pts) == 8:  # Enter
            break
        if k == ord('u') and pts:
            pts.pop()
        if k == ord('r'):
            pts.clear()
        if k == 27:  # Esc
            raise SystemExit("Cancelled by user")
    cv2.destroyWindow(WIN)
    return np.float32(pts)


def target_quads():
    """生成左右镜片在正视图中的目标矩形四点（顺时针：外上、外下、内下、内上）"""
    gap = W_OUT * GAP_RATIO
    lens_w = (W_OUT - gap) * 0.5 * LENS_W_RATIO
    lens_h = H_OUT * LENS_H_RATIO
    cy = H_OUT * 0.52
    cxL = W_OUT * 0.5 - (lens_w + gap) / 2.0
    cxR = W_OUT * 0.5 + (lens_w + gap) / 2.0

    def quad(cx, cy, ww, hh):
        return np.float32([
            [cx - ww/2, cy - hh/2],  # 外上
            [cx - ww/2, cy + hh/2],  # 外下
            [cx + ww/2, cy + hh/2],  # 内下
            [cx + ww/2, cy - hh/2],  # 内上
        ])
    return quad(cxL, cy, lens_w, lens_h), quad(cxR, cy, lens_w, lens_h)


def rectify_two_lenses(img, pts8):
    """分别对两只镜片做透视矫正并合并到画布"""
    quadL, quadR = target_quads()
    srcL, srcR = pts8[:4], pts8[4:]
    M_L = cv2.getPerspectiveTransform(srcL, quadL)
    M_R = cv2.getPerspectiveTransform(srcR, quadR)

    canvas = np.zeros((H_OUT, W_OUT, 3), np.uint8)
    ones = np.ones(img.shape[:2], np.uint8) * 255
    warpL = cv2.warpPerspective(img,  M_L, (W_OUT, H_OUT))
    warpR = cv2.warpPerspective(img,  M_R, (W_OUT, H_OUT))
    maskL = cv2.warpPerspective(ones, M_L, (W_OUT, H_OUT))
    maskR = cv2.warpPerspective(ones, M_R, (W_OUT, H_OUT))
    canvas = np.where(maskL[..., None] > 0, warpL, canvas)
    canvas = np.where(maskR[..., None] > 0, warpR, canvas)
    return canvas


def extract_frame_rgba(front_bgr):
    """从正视图中提取镜框并输出 RGBA"""
    bgr = front_bgr
    blur = cv2.bilateralFilter(bgr, 7, 60, 60)
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

    # 基础边缘/暗色
    edges = cv2.Canny(gray, 40, 120)
    dark  = (gray < 175).astype(np.uint8) * 255
    mask  = cv2.bitwise_or(edges, dark)

    # 清理 + 连通
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)), 1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)), 2)

    # 是否保留镜腿：不保留就裁掉左右边缘
    if not KEEP_TEMPLES:
        trim = int(W_OUT * TEMPLE_TRIM)
        mask[:, :trim]  = 0
        mask[:, -trim:] = 0

    if OUTLINE_ONLY:
        outer = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(RIM_THICK,RIM_THICK)), 1)
        inner = cv2.erode (mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(RIM_THICK,RIM_THICK)), 1)
        rim = cv2.subtract(outer, inner)
        alpha = cv2.GaussianBlur(rim, (3,3), 0)
    else:
        alpha = cv2.GaussianBlur(mask, (3,3), 0)

    bgra = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
    for c in range(3):
        bgra[:,:,c] = np.where(alpha > 0, bgr[:,:,c], 0)
    bgra[:,:,3] = alpha
    return bgra


def main():
    if len(sys.argv) < 2:
        print("用法: python rectify_and_cut.py path/to/glasses.jpg")
        sys.exit(1)

    inp = sys.argv[1]
    img0 = cv2.imread(inp)
    assert img0 is not None, "无法读取图像: "+inp

    # 等比压到不小于 1600 宽再交互
    h0, w0 = img0.shape[:2]
    if w0 < 1200:
        scale = 1600.0 / max(w0, 1600)
    else:
        scale = 1600.0 / w0 if w0 > 2000 else 1.0
    img = cv2.resize(img0, (int(w0*scale), int(h0*scale)), interpolation=cv2.INTER_AREA)

    global H_OUT  # 允许根据输入适当调整画布高度
    H_OUT = int(H_OUT)

    pts8 = pick_8_points(img)
    front = rectify_two_lenses(img, pts8)
    bgra  = extract_frame_rgba(front)

    out = pathlib.Path(inp).with_name(OUT_NAME).as_posix()
    cv2.imwrite(out, bgra)
    print("已保存:", out)
    print("建议锚点（PNG 像素坐标）：")
    # 铰链大致位置：左右画布 15%/85% 处的镜框中线（给个起步参考）
    y_mid = int(H_OUT * 0.52)
    print("SRC_L =", (int(W_OUT*0.15), y_mid),
          " SRC_R =", (int(W_OUT*0.85), y_mid),
          " SRC_N =", (int(W_OUT*0.50), int(H_OUT*0.46)))

if __name__ == "__main__":
    main()
