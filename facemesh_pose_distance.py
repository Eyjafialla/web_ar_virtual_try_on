# facemesh_pose_distance.py
import cv2
import mediapipe as mp
import numpy as np
import json, os, time
# ---- 放到文件顶部：工具函数 ----
import math

# def draw_rotated_box(img, center, size, angle_deg, fill_color=(0, 200, 255), alpha=0.25, border=2, border_color=(255,255,255)):
#     """
#     在 img 上画一个旋转矩形（带透明填充）。
#     center=(cx,cy), size=(w,h), angle 以度为单位（与 cv2.boxPoints 一致）。
#     """
#     overlay = img.copy()
#     rect = (tuple(map(float, center)), tuple(map(float, size)), float(angle_deg))
#     box = cv2.boxPoints(rect).astype(np.int32)
#     cv2.fillConvexPoly(overlay, box, fill_color)
#     # 透明融合
#     cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
#     # 外边框
#     cv2.polylines(img, [box], True, border_color, border, lineType=cv2.LINE_AA)
#     return box  # 如需进一步使用四点坐标

def draw_rotated_box(img, center, size, angle_deg,
                     fill_color=(255, 0, 255),  # 亮紫色
                     alpha=0.60,               # 提高透明度，明显一点
                     border=3, border_color=(0, 255, 255)):
    overlay = img.copy()
    rect = (tuple(map(float, center)), tuple(map(float, size)), float(angle_deg))
    box = cv2.boxPoints(rect).astype(np.int32)
    cv2.fillConvexPoly(overlay, box, fill_color)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    cv2.polylines(img, [box], True, border_color, border, lineType=cv2.LINE_AA)
    return box


# ---- 摄像头参数（你机子 index=1 正常） ----
INDEX   = 1
BACKEND = cv2.CAP_DSHOW      # 不稳可换 cv2.CAP_MSMF
REQ_W, REQ_H = 1280, 720

# ---- FaceMesh 关键点 ----
LM_LEFT_OUTER  = 263
LM_RIGHT_OUTER = 33
LM_NOSE_BRIDGE = 168

# ---- 标定：在 KNOWN_DIST_CM 处按 C 键 ----
KNOWN_DIST_CM = 60.0
CALIB_FILE = "distance_calib.json"   # 保存/读取常量 K

# ---- 平滑参数 ----
EMA_A_POINTS = 0.8   # 坐标平滑
EMA_A_VALUES = 0.8   # 数值平滑

def ema(prev, curr, a):
    return curr if prev is None else a*prev + (1-a)*curr

def to_px(lm, w, h):
    return np.array([lm.x * w, lm.y * h], dtype=np.float32)

def yaw_pitch_roll(pL, pR, pN):
    """
    基于2D的近似：
    - roll: 眼线角度
    - 把鼻点相对眼线中点的向量旋转到眼线坐标系中，得到 (x', y')
      yaw  ~ atan2(x', 0.5*|LR|)
      pitch~ atan2(-y', |LR|)
    """
    v = pR - pL
    d = np.linalg.norm(v) + 1e-6
    roll = np.degrees(np.arctan2(v[1], v[0]))
    mid = (pL + pR) * 0.5
    n = pN - mid

    theta = np.arctan2(v[1], v[0])
    c, s = np.cos(-theta), np.sin(-theta)
    Rm = np.array([[c, -s],[s, c]], dtype=np.float32)
    n_aligned = Rm @ n

    yaw   = np.degrees(np.arctan2(n_aligned[0], 0.5*d))
    pitch = np.degrees(np.arctan2(-n_aligned[1], d))
    return yaw, pitch, roll, d

# ---- 读取/保存标定常量 K（K = d_pixels * distance_cm） ----
def load_K():
    if os.path.exists(CALIB_FILE):
        try:
            with open(CALIB_FILE, "r", encoding="utf-8") as f:
                return float(json.load(f)["K"])
        except Exception:
            pass
    return None

def save_K(K):
    with open(CALIB_FILE, "w", encoding="utf-8") as f:
        json.dump({"K": float(K)}, f)

K = load_K()

# ---- 初始化 ----
cap = cv2.VideoCapture(INDEX, BACKEND)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, REQ_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, REQ_H)
if not cap.isOpened():
    raise RuntimeError("相机打开失败，试试改 BACKEND 或 INDEX。")

mp_fm = mp.solutions.face_mesh
pts_prev = None
vals_prev = None
info_text = ""
info_t_end = 0

def toast(msg, dur=1.5):
    global info_text, info_t_end
    info_text = msg
    info_t_end = time.time() + dur

with mp_fm.FaceMesh(
    max_num_faces=1, refine_landmarks=True,
    min_detection_confidence=0.5, min_tracking_confidence=0.5
) as fm:
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = fm.process(rgb)

        if res.multi_face_landmarks:
            lm = res.multi_face_landmarks[0].landmark
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
            v = pR_s - pL_s
            d = float(np.linalg.norm(v)) + 1e-6
            theta = math.degrees(math.atan2(v[1], v[0]))
            mid = (pL_s + pR_s) * 0.5

            rad = math.radians(theta)
            normal = np.array([-math.sin(rad), math.cos(rad)], dtype=np.float32)
            if normal[1] < 0: normal = -normal
            center = mid + normal * (0.12 * d)

            box_w = 1.85 * d
            box_h = 0.70 * d
            # draw_rotated_box(frame, (float(center[0]), float(center[1])),
            #                  (float(box_w), float(box_h)), theta)
            # 画矩形（返回四个角点）
            corners = draw_rotated_box(frame,
                           (float(center[0]), float(center[1])),
                           (float(box_w), float(box_h)),
                           theta)

            for (x, y) in corners:
                cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 255), -1)

            yaw, pitch, roll, d = yaw_pitch_roll(pL_s, pR_s, pN_s)

            # 视距（cm）：distance = K / d
            dist_cm = None
            if K is not None:
                dist_cm = (K / max(d, 1e-6))

            # 数值EMA
            cur_vals = np.array([
                yaw, pitch, roll,
                0.0 if dist_cm is None else dist_cm
            ], dtype=np.float32)
            if vals_prev is None:
                vals_s = cur_vals
            else:
                vals_s = ema(vals_prev, cur_vals, EMA_A_VALUES)
            vals_prev = vals_s

            yaw_s, pitch_s, roll_s, dist_s = vals_s

            # ---- 绘制可视化 ----
            for P, label in [(pL_s, "263"), (pR_s, "33"), (pN_s, "168")]:
                cv2.circle(frame, (int(P[0]), int(P[1])), 4, (0,255,0), -1)
                cv2.putText(frame, label, (int(P[0])+6, int(P[1])-6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2, cv2.LINE_AA)
            cv2.line(frame, (int(pL_s[0]), int(pL_s[1])), (int(pR_s[0]), int(pR_s[1])), (255,255,255), 2)

            # 左上角数据显示
            y0 = 30
            step = 28
            cv2.putText(frame, f"Yaw  : {yaw_s:+.1f} deg",   (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA); y0+=step
            cv2.putText(frame, f"Pitch: {pitch_s:+.1f} deg", (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA); y0+=step
            cv2.putText(frame, f"Roll : {roll_s:+.1f} deg",  (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA); y0+=step
            if K is None:
                cv2.putText(frame, f"Distance: (press 'C' at {KNOWN_DIST_CM:.0f} cm to calibrate)",
                            (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2, cv2.LINE_AA)
            else:
                cv2.putText(frame, f"Distance: {dist_s:5.1f} cm",
                            (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)

        # 临时提示
        if time.time() < info_t_end:
            cv2.putText(frame, info_text, (10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50,220,50), 2, cv2.LINE_AA)

        cv2.imshow("FaceMesh pose & distance", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')): break
        if key in (ord('c'), ord('C')) and res.multi_face_landmarks:
            # 用当前 d 与已知距离标定 K
            # 注意：请把脸保持正对摄像头，在距离摄像头 KNOWN_DIST_CM 处按 C
            K = d * KNOWN_DIST_CM
            save_K(K)
            toast(f"Calibrated: K={K:.1f} (at {KNOWN_DIST_CM:.0f} cm)")

cap.release()
cv2.destroyAllWindows()
