# facemesh_ar_glasses.py
# -*- coding: utf-8 -*-
# 依赖: pip install opencv-python mediapipe
import collections
import os, time, math, json
import cv2
import numpy as np

# ---------- (可选) 静音 Mediapipe/TF 的日志 ----------
os.environ.setdefault("GLOG_minloglevel", "3")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
try:
    import absl.logging as _absl
    _absl.set_verbosity(_absl.ERROR)
except Exception:
    pass

import mediapipe as mp

# —— 眼镜贴图微调参数 ——（可用热键现场调）
GLASSES_SCALE   = 1.00  # 整体等比缩放，>1 放大
GLASSES_SCALE_X = 1.00  # 沿“眼线方向”的宽度缩放
GLASSES_SCALE_Y = 1.00  # 沿“法线方向”(上下高度)的缩放
GLASSES_OFFSET_N = 0.00 # 沿法线偏移（乘以眼距 d），正值向下
GLASSES_SHIFT_T  = 0.00 # 沿眼线切向平移（乘以眼距 d），正值往右

# ---------- 基本配置 ----------
CAM_INDEX   = 1
CAM_BACKEND = cv2.CAP_DSHOW
REQ_W, REQ_H = 1280, 720

SHOW_LANDMARKS = False
SHOW_HUD       = True

# ---------- FaceMesh 关键点 ----------
LM_LEFT_OUTER  = 263              # 左眼外角（flip(1) 后使用这个为“左”）
LM_RIGHT_OUTER = 33               # 右眼外角
LM_NOSE_BRIDGE = 168              # 鼻梁

# ---------- 平滑参数 ----------
EMA_A_POINTS = 0.80               # 坐标平滑 (越大越稳)
EMA_A_VALUES = 0.85               # 数值平滑 (越大越稳)

# ---------- 距离标定 ----------
KNOWN_DIST_CM = 60.0
CALIB_FILE = "distance_calib.json"

# ---------- 贴图: PNG (可带或不带透明通道) ----------
OVERLAY_PATH = "Frame_A.png"

# 当贴图虽然是 RGBA，但 alpha 基本全 255（镜片是白块）时，强制用 edge 重建透明通道
FORCE_EDGE_MASK = True

def build_alpha_edge(img_bgr, edge_lo=60, edge_hi=160, dilate=5, feather=2,
                     band_top=0.20, band_bot=0.90):
    """从白底/浅底眼镜图中提取边缘作为 alpha，去掉镜片白块。"""
    h, w = img_bgr.shape[:2]
    gray  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, edge_lo, edge_hi)
    if dilate > 0:
        edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(dilate,dilate)), 1)
    band = np.zeros_like(edges)
    y0, y1 = int(band_top*h), int(band_bot*h)
    band[y0:y1, :] = 255
    edges = cv2.bitwise_and(edges, band)
    alpha = edges
    if feather > 0:
        alpha = cv2.GaussianBlur(alpha, (0,0), feather)
    rgba = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2BGRA)
    rgba[:,:,3] = alpha
    return rgba

# 读取贴图并确保透明通道可用
tmp = cv2.imread(OVERLAY_PATH, cv2.IMREAD_UNCHANGED)
print("overlay", OVERLAY_PATH, "shape=", None if tmp is None else tmp.shape)
if tmp is None:
    raise RuntimeError("找不到贴图: %s" % OVERLAY_PATH)

if tmp.ndim == 3 and tmp.shape[2] == 4:
    a = tmp[:,:,3]
    opaque_ratio = (a > 250).mean()
    if FORCE_EDGE_MASK and opaque_ratio > 0.98:
        overlay_rgba = build_alpha_edge(tmp[:,:,:3])
    else:
        overlay_rgba = tmp
elif tmp.ndim == 3 and tmp.shape[2] == 3:
    overlay_rgba = build_alpha_edge(tmp)
else:
    raise RuntimeError("贴图通道数不支持: %s" % (tmp.shape,))

H_ov, W_ov = overlay_rgba.shape[:2]

# 贴图里的三个“锚点”(像素坐标) —— 对应: 263(左外), 33(右外), 168(鼻梁)
# 不对齐就微调
SRC_L = (int(0.15 * W_ov), int(0.52 * H_ov))
SRC_R = (int(0.85 * W_ov), int(0.52 * H_ov))
SRC_N = (int(0.50 * W_ov), int(0.46 * H_ov))
# 若左右反：交换 SRC_L/SRC_R 或 cv2.flip(overlay_rgba,1)

# ---------- 工具函数 ----------
def to_px(lm, w, h):
    return np.array([lm.x * w, lm.y * h], dtype=np.float32)

def ema(prev, curr, a):
    return curr if prev is None else (a * prev + (1 - a) * curr)

def yaw_pitch_roll(pL, pR, pN):
    """基于2D近似: 返回 yaw/pitch/roll(度)、两眼像素距 d"""
    v = pR - pL
    d = float(np.linalg.norm(v)) + 1e-6
    roll = math.degrees(math.atan2(float(v[1]), float(v[0])))
    mid = (pL + pR) * 0.5
    n = pN - mid
    theta = math.atan2(float(v[1]), float(v[0]))
    c, s = math.cos(-theta), math.sin(-theta)
    n_aligned = np.array([c * n[0] - s * n[1], s * n[0] + c * n[1]], dtype=np.float32)
    yaw   = math.degrees(math.atan2(float(n_aligned[0]), 0.5 * d))
    pitch = math.degrees(math.atan2(float(-n_aligned[1]), d))
    return yaw, pitch, roll, d

# 旧的仿射函数仍保留（不再使用），以便需要时切回
def overlay_affine_rgba(dst_bgr, rgba, src_tri, dst_tri):
    src_tri = np.float32(src_tri); dst_tri = np.float32(dst_tri)
    M = cv2.getAffineTransform(src_tri, dst_tri)
    H, W = dst_bgr.shape[:2]
    d_src = float(np.linalg.norm(src_tri[1] - src_tri[0]) + 1e-6)
    d_dst = float(np.linalg.norm(dst_tri[1] - dst_tri[0]) + 1e-6)
    s = d_dst / d_src
    interp = cv2.INTER_LANCZOS4 if s > 1.02 else cv2.INTER_AREA
    warped = cv2.warpAffine(rgba, M, (W, H), flags=interp, borderMode=cv2.BORDER_TRANSPARENT)
    b, g, r, a = cv2.split(warped)
    fg = cv2.merge([b, g, r]).astype(np.float32)
    if s > 1.02:
        blur = cv2.GaussianBlur(fg, (0, 0), 0.9)
        fg = cv2.addWeighted(fg, 1.28, blur, -0.28, 0)
    mask = (a.astype(np.float32) / 255.0)[..., None]
    dst_bgr[:] = (dst_bgr*(1-mask) + fg*mask).astype(np.uint8)

def load_K():
    if os.path.exists(CALIB_FILE):
        try:
            with open(CALIB_FILE, "r", encoding="utf-8") as f:
                return float(json.load(f)["K"])
        except Exception:
            return None
    return None

def save_K(K):
    with open(CALIB_FILE, "w", encoding="utf-8") as f:
        json.dump({"K": float(K)}, f)

# ---------- 相机 ----------
cap = cv2.VideoCapture(CAM_INDEX, CAM_BACKEND)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, REQ_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, REQ_H)
if not cap.isOpened():
    raise RuntimeError("相机打开失败，请检查 index/backend，或关闭占用相机的软件。")

# ---------- FaceMesh ----------
mp_fm = mp.solutions.face_mesh
fm = mp_fm.FaceMesh(max_num_faces=1, refine_landmarks=True,
                    min_detection_confidence=0.5, min_tracking_confidence=0.5)

# ---------- 运行时状态 ----------
pts_prev = None
vals_prev = None
K = load_K()
print("Loaded K:", K)

last = time.perf_counter()
fps_ema = 0.0
fps_win = collections.deque(maxlen=60)
show_fps = True

try:
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # --- FPS统计 ---
        now = time.perf_counter()
        dt = now - last; last = now
        if dt > 0:
            fps_inst = 1.0 / dt
            fps_ema = fps_inst if fps_ema == 0 else (0.9*fps_ema + 0.1*fps_inst)
            fps_win.append(fps_inst)
            fps_avg = sum(fps_win) / len(fps_win)
        else:
            fps_avg = fps_ema

        frame = cv2.flip(frame, 1)  # 自拍视图
        h, w = frame.shape[:2]

        # 推理
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = fm.process(rgb)

        if res.multi_face_landmarks:
            lm = res.multi_face_landmarks[0].landmark

            # 三点(像素)
            pL = to_px(lm[LM_LEFT_OUTER],  w, h)
            pR = to_px(lm[LM_RIGHT_OUTER], w, h)
            pN = to_px(lm[LM_NOSE_BRIDGE], w, h)

            # 坐标EMA
            if pts_prev is None:
                pL_s, pR_s, pN_s = pL, pR, pN
            else:
                pL_s = ema(pts_prev[0], pL, EMA_A_POINTS)
                pR_s = ema(pts_prev[1], pR, EMA_A_POINTS)
                pN_s = ema(pts_prev[2], pN, EMA_A_POINTS)
            pts_prev = (pL_s, pR_s, pN_s)

            # 左右兜底交换（避免 roll≈180° 带来“整体斜”）
            if pL_s[0] > pR_s[0]:
                pL_s, pR_s = pR_s, pL_s

            # 角度/距离（仅用于 HUD）
            yaw, pitch, roll, d_px = yaw_pitch_roll(pL_s, pR_s, pN_s)
            dist_cm = (K / d_px) if K is not None else None
            cur_vals = np.array([yaw, pitch, roll, 0.0 if dist_cm is None else dist_cm], np.float32)
            vals_s = cur_vals if vals_prev is None else ema(vals_prev, cur_vals, EMA_A_VALUES)
            vals_prev = vals_s
            yaw_s, pitch_s, roll_s, dist_s = map(float, vals_s)

            # ---- 无剪切方式：旋转 + 各向异性缩放 + 中心对齐 ----
            v = pR_s - pL_s
            d = float(np.linalg.norm(v)) + 1e-6
            u = v / d                                  # 眼线方向
            n = np.array([-u[1], u[0]], np.float32)    # 法向（图像坐标下方为正）
            if n[1] < 0:
                n = -n
            ctr = (pL_s + pR_s) * 0.5                  # 双眼中点

            # 贴图内左右锚点的中点当作“贴图中心”
            ovL = np.array(SRC_L, np.float32)
            ovR = np.array(SRC_R, np.float32)
            ov_ctr = (ovL + ovR) * 0.5
            ov_dist = float(np.linalg.norm(ovR - ovL)) + 1e-6

            # 旋转角与各向异性缩放
            theta = math.atan2(float(u[1]), float(u[0]))  # 弧度
            base  = d / ov_dist
            sx = base * GLASSES_SCALE * GLASSES_SCALE_X   # 沿 u 方向
            sy = base * GLASSES_SCALE * GLASSES_SCALE_Y   # 沿 n 方向

            # 目标中心（加切向/法向偏移）
            dst_ctr = ctr + (GLASSES_SHIFT_T * d) * u + (GLASSES_OFFSET_N * d) * n

            # 组合“各向异性缩放 + 旋转”的 2x3 仿射矩阵（无剪切）
            c, s = math.cos(theta), math.sin(theta)
            M = np.array([[ sx*c, -sy*s, 0.0 ],
                          [ sx*s,  sy*c, 0.0 ]], dtype=np.float32)
            # 平移使贴图中心落到目标中心
            t = dst_ctr - (M[:, :2] @ ov_ctr)
            M[:, 2] = t

            # warp 整张 RGBA 并 alpha 混合（放大用 Lanczos，缩小时用 AREA）
            Hf, Wf = frame.shape[:2]
            up_ratio = max(sx, sy)
            interp = cv2.INTER_LANCZOS4 if up_ratio > 1.02 else cv2.INTER_AREA
            warped = cv2.warpAffine(overlay_rgba, M, (Wf, Hf),
                                    flags=interp, borderMode=cv2.BORDER_TRANSPARENT)
            b,g,r,a = cv2.split(warped)
            fg   = cv2.merge([b,g,r]).astype(np.float32)
            # 轻锐化（只在放大时）
            if up_ratio > 1.02:
                blur = cv2.GaussianBlur(fg, (0,0), 0.9)
                fg = cv2.addWeighted(fg, 1.28, blur, -0.28, 0)
            mask = (a.astype(np.float32)/255.0)[...,None]
            frame[:] = (frame*(1-mask) + fg*mask).astype(np.uint8)

            # (可选) 调试可视化
            if SHOW_LANDMARKS:
                for P, label in [(pL_s, "263"), (pR_s, "33"), (pN_s, "168")]:
                    cv2.circle(frame, (int(P[0]), int(P[1])), 4, (0,255,0), -1)
                    cv2.putText(frame, label, (int(P[0])+6, int(P[1])-6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2, cv2.LINE_AA)
                cv2.line(frame, (int(pL_s[0]), int(pL_s[1])),
                                (int(pR_s[0]), int(pR_s[1])), (255,255,255), 2)

            # HUD
            if SHOW_HUD:
                y0, step = 30, 28
                cv2.putText(frame, f"Yaw  : {yaw_s:+.1f} deg",   (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.8,(255,255,255),2,cv2.LINE_AA); y0+=step
                cv2.putText(frame, f"Pitch: {pitch_s:+.1f} deg", (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.8,(255,255,255),2,cv2.LINE_AA); y0+=step
                cv2.putText(frame, f"Roll : {roll_s:+.1f} deg",  (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.8,(255,255,255),2,cv2.LINE_AA); y0+=step
                if K is None:
                    cv2.putText(frame, f"Distance: (press 'C' at {KNOWN_DIST_CM:.0f} cm to calibrate)",
                                (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(200,200,200),2,cv2.LINE_AA)
                else:
                    cv2.putText(frame, f"Distance: {dist_s:5.1f} cm",
                                (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.8,(255,255,255),2,cv2.LINE_AA)

            if show_fps:
                info = f"{w}x{h} | FPS {fps_ema:.1f} (avg {fps_avg:.1f})"
                (tw, th), _ = cv2.getTextSize(info, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                cv2.rectangle(frame, (w - tw - 20, 10), (w - 10, 10 + th + 10), (0, 0, 0), -1)
                cv2.putText(frame, info, (w - tw - 15, 10 + th),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)

        cv2.imshow("FaceMesh AR Glasses", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):
            break
        elif key == ord('s'):
            fname = f"snapshot_{int(time.time())}.png"
            cv2.imwrite(fname, frame); print("[saved]", fname)
        elif key in (ord('c'), ord('C')) and 'd_px' in locals():
            K = d_px * KNOWN_DIST_CM
            save_K(K)
            print(f"[calibrated] K={K:.1f} at {KNOWN_DIST_CM:.0f} cm")
        elif key == ord('f'):
            show_fps = not show_fps

        # —— 尺寸热键 ——
        elif key == ord(']'):  GLASSES_SCALE = min(GLASSES_SCALE + 0.02, 2.0);  print("SCALE:", GLASSES_SCALE)
        elif key == ord('['):  GLASSES_SCALE = max(GLASSES_SCALE - 0.02, 0.6);  print("SCALE:", GLASSES_SCALE)
        # 宽度（沿眼线方向）
        elif key == ord('0'):  GLASSES_SCALE_X = min(GLASSES_SCALE_X + 0.02, 2.0); print("SX:", GLASSES_SCALE_X)
        elif key == ord('9'):  GLASSES_SCALE_X = max(GLASSES_SCALE_X - 0.02, 0.6); print("SX:", GLASSES_SCALE_X)
        # 高度（沿法线方向）
        elif key == ord("'"):  GLASSES_SCALE_Y = min(GLASSES_SCALE_Y + 0.02, 2.0); print("SY:", GLASSES_SCALE_Y)
        elif key == ord(';'):  GLASSES_SCALE_Y = max(GLASSES_SCALE_Y - 0.02, 0.6); print("SY:", GLASSES_SCALE_Y)
        # 上下平移（法线）
        elif key == ord('='):  GLASSES_OFFSET_N += 0.01;  print("OFF_N:", GLASSES_OFFSET_N)
        elif key == ord('-'):  GLASSES_OFFSET_N -= 0.01;  print("OFF_N:", GLASSES_OFFSET_N)
        # 左右平移（沿眼线）
        elif key == ord('.'):  GLASSES_SHIFT_T += 0.01;   print("SHIFT_T:", GLASSES_SHIFT_T)
        elif key == ord(','):  GLASSES_SHIFT_T -= 0.01;   print("SHIFT_T:", GLASSES_SHIFT_T)

finally:
    fm.close()
    cap.release()
    cv2.destroyAllWindows()


