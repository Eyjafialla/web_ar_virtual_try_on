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
GLASSES_SCALE_Y = 0.75  # 沿“法线方向”(上下高度)的缩放
GLASSES_OFFSET_N = 0.00 # 沿法线偏移（乘以眼距 d），正值向下
GLASSES_SHIFT_T  = 0.00 # 沿眼线切向平移（乘以眼距 d），正值往右

# ---------- 基本配置 ----------
CAM_INDEX   = 1                   # 你的机器 index=1 正常
CAM_BACKEND = cv2.CAP_DSHOW       # 不稳可换 cv2.CAP_MSMF
REQ_W, REQ_H = 1280, 720

SHOW_LANDMARKS = False            # 调试可视化(绿点/编号/白线)
SHOW_HUD       = True             # 左上角显示角度/距离

# ---------- FaceMesh 关键点 ----------
LM_LEFT_OUTER  = 263              # 左眼外角
LM_RIGHT_OUTER = 33               # 右眼外角
LM_NOSE_BRIDGE = 168              # 鼻梁

# ---------- 平滑参数 ----------
EMA_A_POINTS = 0.80               # 坐标平滑 (越大越稳)
EMA_A_VALUES = 0.85               # 数值平滑 (越大越稳)

# ---------- 距离标定 ----------
KNOWN_DIST_CM = 60.0              # 你实测的标定距离(可改)
CALIB_FILE = "distance_calib.json"

# ---------- 贴图: PNG (必须 RGBA) ----------
# 放一张带透明通道的眼镜 PNG 到本文件同目录; 改成你的文件名即可
OVERLAY_PATH = "Frame_A.png"      
overlay_rgba = cv2.imread(OVERLAY_PATH, cv2.IMREAD_UNCHANGED)
print("overlay", OVERLAY_PATH, "shape=", None if overlay_rgba is None else overlay_rgba.shape)
if overlay_rgba is None or overlay_rgba.ndim != 3 or overlay_rgba.shape[2] != 4:
    raise RuntimeError("找不到贴图或不是 RGBA(4通道) PNG: %s" % OVERLAY_PATH)

H_ov, W_ov = overlay_rgba.shape[:2]

# 贴图里的三个“锚点”(像素坐标) —— 首次可按大小估一下, 跑起来再微调即可
# 对应顺序: 263(左眼外角), 33(右眼外角), 168(鼻梁)
# 先给一组通用估值(居中矩形框两侧+鼻梁稍上)，如果对不齐就改这三个数
SRC_L = (int(0.15 * W_ov), int(0.52 * H_ov))
SRC_R = (int(0.85 * W_ov), int(0.52 * H_ov))
SRC_N = (int(0.50 * W_ov), int(0.46 * H_ov))
# SRC_L, SRC_R, SRC_N = (105,276), (1123,276), (530,282)
# SRC_L = (175, 217)   # = (0.15*W, 0.52*H)
# SRC_R = (993, 217)   # = (0.85*W, 0.52*H)
# SRC_N = (584, 190)   # = (0.50*W, 0.46*H)
# 如果左右反了：把 SRC_L / SRC_R 互换，或 overlay_rgba = cv2.flip(overlay_rgba, 1)

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

def overlay_affine_rgba(dst_bgr, rgba, src_tri, dst_tri):
    """三点仿射 + 自适应插值 + 可选锐化"""
    src_tri = np.float32(src_tri); dst_tri = np.float32(dst_tri)
    M = cv2.getAffineTransform(src_tri, dst_tri)

    H, W = dst_bgr.shape[:2]
    # 估计缩放比例（用两眼距离）
    d_src = float(np.linalg.norm(src_tri[1] - src_tri[0]) + 1e-6)
    d_dst = float(np.linalg.norm(dst_tri[1] - dst_tri[0]) + 1e-6)
    s = d_dst / d_src

    interp = cv2.INTER_LANCZOS4 if s > 1.02 else cv2.INTER_AREA
    warped = cv2.warpAffine(
        rgba, M, (W, H),
        flags=interp, borderMode=cv2.BORDER_TRANSPARENT
    )
    b, g, r, a = cv2.split(warped)
    warped_bgr = cv2.merge([b, g, r]).astype(np.float32)

    # 放大时轻锐化一遍，增加边缘清晰度
    if s > 1.02:
        blur = cv2.GaussianBlur(warped_bgr, (0, 0), 0.9)
        warped_bgr = cv2.addWeighted(warped_bgr, 1.28, blur, -0.28, 0)

    mask = (a.astype(np.float32) / 255.0)
    mask3 = cv2.merge([mask, mask, mask])
    bg = dst_bgr.astype(np.float32)

    out = bg * (1.0 - mask3) + warped_bgr * mask3
    dst_bgr[:] = np.clip(out, 0, 255).astype(np.uint8)


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
K = load_K()           # 距离标定常量
print("Loaded K:", K)
# 采集循环
last = time.perf_counter()
fps_ema = 0.0                     # 指数平滑
fps_win = collections.deque(maxlen=60)  # 1 秒窗口(60fps下)
show_fps = True

try:
    while True:
        ok, frame = cap.read()
        if not ok:
            break
            # --- FPS 统计（就放在 cap.read() 后面）---
        now = time.perf_counter()
        dt = now - last
        last = now
        if dt > 0:
            fps_inst = 1.0 / dt
            fps_ema = fps_inst if fps_ema == 0 else (0.9*fps_ema + 0.1*fps_inst)
            fps_win.append(fps_inst)
            fps_avg = sum(fps_win) / len(fps_win)
        else:
            fps_inst = fps_avg = fps_ema
        frame = cv2.flip(frame, 1)  # 自拍视图
        h, w = frame.shape[:2]

        # 推理
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = fm.process(rgb)

        if res.multi_face_landmarks:
            lm = res.multi_face_landmarks[0].landmark

            # 取三点(像素)
            pL = to_px(lm[LM_LEFT_OUTER],  w, h)
            pR = to_px(lm[LM_RIGHT_OUTER], w, h)
            pN = to_px(lm[LM_NOSE_BRIDGE], w, h)

            # 坐标平滑
            if pts_prev is None:
                pL_s, pR_s, pN_s = pL, pR, pN
            else:
                pL_s = ema(pts_prev[0], pL, EMA_A_POINTS)
                pR_s = ema(pts_prev[1], pR, EMA_A_POINTS)
                pN_s = ema(pts_prev[2], pN, EMA_A_POINTS)
            pts_prev = (pL_s, pR_s, pN_s)

            # 角度/距离
            yaw, pitch, roll, d_px = yaw_pitch_roll(pL_s, pR_s, pN_s)
            dist_cm = (K / d_px) if K is not None else None

            cur_vals = np.array([yaw, pitch, roll, 0.0 if dist_cm is None else dist_cm], np.float32)
            vals_s = cur_vals if vals_prev is None else ema(vals_prev, cur_vals, EMA_A_VALUES)
            vals_prev = vals_s
            yaw_s, pitch_s, roll_s, dist_s = map(float, vals_s)

            # 贴 PNG 眼镜
            # dst_tri = [(float(pL_s[0]), float(pL_s[1])),
            #            (float(pR_s[0]), float(pR_s[1])),
            #            (float(pN_s[0]), float(pN_s[1]))]
            # src_tri = [SRC_L, SRC_R, SRC_N]
            # overlay_affine_rgba(frame, overlay_rgba, src_tri, dst_tri)
            # ---- 计算各向异性缩放 + 平移后的三点 ----
            v = pR_s - pL_s
            d = float(np.linalg.norm(v)) + 1e-6
            u = v / d                                    # 眼线方向(单位向量)
            n = np.array([-u[1], u[0]], dtype=np.float32)  # 法线（向下）
            if n[1] < 0: n = -n
            ctr = (pL_s + pR_s) * 0.5                     # 双眼中点

            def tf(P):
                # 转到 (u, n) 基
                rel = P - ctr
                x = float(np.dot(rel, u))
                y = float(np.dot(rel, n))
                # 先等比，再各向异性缩放
                x *= (GLASSES_SCALE * GLASSES_SCALE_X)
                y *= (GLASSES_SCALE * GLASSES_SCALE_Y)
                # 切向/法向平移
                x += GLASSES_SHIFT_T  * d
                y += GLASSES_OFFSET_N * d
                # 回到像素坐标
                return ctr + x * u + y * n

            pL_d, pR_d, pN_d = tf(pL_s), tf(pR_s), tf(pN_s)
            dst_tri = [(float(pL_d[0]), float(pL_d[1])),
                       (float(pR_d[0]), float(pR_d[1])),
                       (float(pN_d[0]), float(pN_d[1]))]
            overlay_affine_rgba(frame, overlay_rgba, [SRC_L, SRC_R, SRC_N], dst_tri)


            # (可选) 调试可视化
            if SHOW_LANDMARKS:
                for P, label in [(pL_s, "263"), (pR_s, "33"), (pN_s, "168")]:
                    cv2.circle(frame, (int(P[0]), int(P[1])), 4, (0,255,0), -1)
                    cv2.putText(frame, label, (int(P[0])+6, int(P[1])-6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2, cv2.LINE_AA)
                cv2.line(frame, (int(pL_s[0]), int(pL_s[1])), (int(pR_s[0]), int(pR_s[1])), (255,255,255), 2)

            # (可选) HUD
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
                h, w = frame.shape[:2]
                info = f"{w}x{h} | FPS {fps_ema:.1f} (avg {fps_avg:.1f})"
                (tw, th), _ = cv2.getTextSize(info, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                cv2.rectangle(frame, (w - tw - 20, 10), (w - 10, 10 + th + 10), (0, 0, 0), -1)
                cv2.putText(frame, info, (w - tw - 15, 10 + th),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)

        cv2.imshow("FaceMesh AR Glasses", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):      # 退出
            break
        elif key in (ord('s'),):       # 截图
            fname = f"snapshot_{int(time.time())}.png"
            cv2.imwrite(fname, frame); print("[saved]", fname)
        elif key in (ord('c'), ord('C')) and 'd_px' in locals():
            # 在已知距离 KNOWN_DIST_CM 处正对屏幕按 C 完成一次标定
            K = d_px * KNOWN_DIST_CM
            save_K(K)
            print(f"[calibrated] K={K:.1f} at {KNOWN_DIST_CM:.0f} cm")
        elif key == ord('f'):           # ← 开关 FPS 显示
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
