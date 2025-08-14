import os, time, math, json, threading
import cv2, numpy as np
from flask import Flask, Response, request, jsonify, send_from_directory
import mediapipe as mp

# ------------ 资源路径 ------------
HERE = os.path.dirname(__file__)
FRAME_DIR = os.path.join(HERE, "frames")
FRAME_FILES = {
    "A": "Frame_A_unfold.png",
    "B": "Frame_B_new_rgba.png",
    "C": "Frame_C_new_rgba.png",
}
DEFAULT_FRAME_ID = "A"

# ------------ 叠加参数（网页可改）------------
state_lock = threading.Lock()
STATE = {
    "frameId": DEFAULT_FRAME_ID,
    "scale": 1.00,
    "scaleX": 1.00,
    "scaleY": 1.00,
    "shiftT": 0.00,        # 沿眼线方向 平移（按眼距 d 的倍数）
    "offsetN": 0.00,       # 沿法线方向 平移（按眼距 d 的倍数）
    "templeRotDeg": 0.0,   # 镜腿外张(+)/内收(-) 小角度旋转（度）
}

# ------------ 工具：读取贴图（确保有 alpha）------------
def read_rgba(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"找不到贴图: {path}")
    if img.ndim == 3 and img.shape[2] == 4:
        return img
    elif img.ndim == 3 and img.shape[2] == 3:
        b,g,r = cv2.split(img)
        a = np.full(b.shape, 255, np.uint8)
        return cv2.merge([b,g,r,a])
    else:
        raise RuntimeError(f"通道数不支持: {path}")

def load_frame_png(frame_id):
    fname = FRAME_FILES.get(frame_id)
    if not fname:
        raise RuntimeError(f"未知 frameId: {frame_id}")
    return read_rgba(os.path.join(FRAME_DIR, fname))

# 当前贴图（原图）与缓存（按 templeRotDeg 预变形后）
overlay_rgba_base = load_frame_png(DEFAULT_FRAME_ID)
overlay_rgba_cached = None
cached_key = (DEFAULT_FRAME_ID, None)

H_ov, W_ov = overlay_rgba_base.shape[:2]

# —— 贴图锚点（按素材可微调一次即可）——
SRC_L = (int(0.18*W_ov), int(0.50*H_ov))
SRC_R = (int(0.82*W_ov), int(0.50*H_ov))

# ------------ FaceMesh ------------
mp_fm = mp.solutions.face_mesh
fm = mp_fm.FaceMesh(max_num_faces=1, refine_landmarks=True,
                    min_detection_confidence=0.5, min_tracking_confidence=0.5)
LM_LEFT_OUTER  = 263
LM_RIGHT_OUTER = 33

def to_px(lm, w, h):
    return np.array([lm.x*w, lm.y*h], np.float32)

# ------------ 稳健的 RGBA 叠加（修复 repeat 报错）------------
def alpha_over(bg, fg):
    """
    在同尺寸 patch 上做标准 alpha over。
    输入:
      bg: HxWx3(BGR) 或 HxWx4(BGRA) uint8
      fg: HxWx4(BGRA) 或 HxWx3(BGR) uint8
    返回:
      HxWx4(BGRA) uint8
    """
    # 统一转 BGRA
    if bg.shape[2] == 3:
        bg_rgba = cv2.cvtColor(bg, cv2.COLOR_BGR2BGRA)
    else:
        bg_rgba = bg.copy()
    if fg.shape[2] == 3:
        fg_rgba = cv2.cvtColor(fg, cv2.COLOR_BGR2BGRA)
    else:
        fg_rgba = fg

    bg_rgba = bg_rgba.astype(np.float32)
    fg_rgba = fg_rgba.astype(np.float32)

    af = fg_rgba[..., 3:4] / 255.0
    ab = bg_rgba[..., 3:4] / 255.0

    out_a   = af + ab * (1.0 - af)                    # (H,W,1)
    out_rgb = fg_rgba[..., :3] * af + bg_rgba[..., :3] * (1.0 - af)

    out = np.concatenate([out_rgb, out_a * 255.0], axis=-1)
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out

# ------------ 贴图空间“镜腿小角度旋转”预变形 ------------
def rotate_roi_rgba(img_rgba, center, angle_deg, radius):
    """围绕 center 对半径约 radius 的方形 ROI 旋转，并 alpha 合成回去。"""
    H, W = img_rgba.shape[:2]
    cx, cy = int(center[0]), int(center[1])
    r = int(radius)
    x0, y0 = max(0, cx-r), max(0, cy-r)
    x1, y1 = min(W, cx+r), min(H, cy+r)
    if x1 <= x0 or y1 <= y0:
        return img_rgba

    roi = img_rgba[y0:y1, x0:x1].copy()                      # BGRA
    Hroi, Wroi = roi.shape[:2]

    M = cv2.getRotationMatrix2D((Wroi/2, Hroi/2), angle_deg, 1.0)
    rot = cv2.warpAffine(
        roi, M, (Wroi, Hroi),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0,0,0,0)
    )
    out_patch = alpha_over(roi, rot)                         # BGRA
    out = img_rgba.copy()
    out[y0:y1, x0:x1] = out_patch
    return out

def prewarp_temples(base_rgba, deg, src_L, src_R):
    """在贴图空间对左右镜腿做小角度外张/内收旋转，返回 BGRA。"""
    if abs(deg) < 1e-3:
        return base_rgba
    H, W = base_rgba.shape[:2]
    offset = int(0.03 * W)   # 往外偏移 3% 宽度更像铰链
    center_L = (max(0, src_L[0]-offset), src_L[1])
    center_R = (min(W-1, src_R[0]+offset), src_R[1])
    radius = int(0.28 * W)

    out = rotate_roi_rgba(base_rgba, center_L, +deg, radius)
    out = rotate_roi_rgba(out,       center_R, -deg, radius)
    return out

# 预变形缓存
def get_overlay_for_render():
    """根据 STATE 返回（可能预变形过的）overlay_rgba。做缓存避免每帧重算。"""
    global overlay_rgba_cached, cached_key
    with state_lock:
        fid  = STATE["frameId"]
        deg  = float(STATE.get("templeRotDeg", 0.0))
        base = overlay_rgba_base
        key  = (fid, round(deg, 3))

    if cached_key == key and overlay_rgba_cached is not None:
        return overlay_rgba_cached

    img = prewarp_temples(base, deg, SRC_L, SRC_R)
    overlay_rgba_cached = img
    cached_key = key
    return overlay_rgba_cached

# ------------ 目标空间无剪切仿射贴图 ------------
def warp_noshear_rgba(rgba, src_L, src_R, dst_ctr, u, n, d, frame_shape,
                      scale=1.0, sx=1.0, sy=1.0):
    Hf, Wf = frame_shape[:2]
    ovL = np.array(src_L, np.float32)
    ovR = np.array(src_R, np.float32)
    ov_ctr  = (ovL + ovR) * 0.5
    ov_dist = float(np.linalg.norm(ovR - ovL)) + 1e-6

    base = d / ov_dist
    sx = base * scale * sx
    sy = base * scale * sy

    theta = math.atan2(float(u[1]), float(u[0]))
    c, s = math.cos(theta), math.sin(theta)
    M = np.array([[ sx*c, -sy*s, 0.0 ],
                  [ sx*s,  sy*c, 0.0 ]], np.float32)
    t = dst_ctr - (M[:, :2] @ ov_ctr)
    M[:, 2] = t

    interp = cv2.INTER_LANCZOS4 if max(sx, sy) > 1.02 else cv2.INTER_AREA
    warped = cv2.warpAffine(rgba, M, (Wf, Hf),
                            flags=interp, borderMode=cv2.BORDER_TRANSPARENT)
    b,g,r,a = cv2.split(warped)
    fg  = cv2.merge([b,g,r]).astype(np.float32)
    mask = (a.astype(np.float32)/255.0)[...,None]
    return fg, mask

# ------------ 摄像头与 MJPEG ------------
CAM_INDEX = int(os.environ.get("CAM_INDEX", "1"))  # 需要用 1 就设置 CAM_INDEX=1
cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

last_jpeg = None
last_frame_ts = 0

# 点位 EMA 平滑，降低抖动
EMA_A = 0.80
prev_L = None
prev_R = None

def generate_mjpeg():
    global last_jpeg, last_frame_ts, prev_L, prev_R
    fps_ema = 0.0
    t_prev = time.perf_counter()

    while True:
        ok, frame = cap.read()
        # 黑帧/抢占兜底
        if not ok or frame is None or frame.mean() < 2.0:
            time.sleep(0.01)
            continue

        frame = cv2.flip(frame, 1)
        H, W = frame.shape[:2]

        # FaceMesh
        res = fm.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if res.multi_face_landmarks:
            lm = res.multi_face_landmarks[0].landmark
            pL = to_px(lm[LM_LEFT_OUTER],  W, H)
            pR = to_px(lm[LM_RIGHT_OUTER], W, H)
            if pL[0] > pR[0]:
                pL, pR = pR, pL

            # EMA 平滑
            if prev_L is None:
                pL_s, pR_s = pL, pR
            else:
                pL_s = EMA_A * prev_L + (1.0 - EMA_A) * pL
                pR_s = EMA_A * prev_R + (1.0 - EMA_A) * pR
            prev_L, prev_R = pL_s, pR_s

            v = pR_s - pL_s
            d = float(np.linalg.norm(v)) + 1e-6
            u = v / d
            n = np.array([-u[1], u[0]], np.float32)
            if n[1] < 0: n = -n
            ctr = (pL_s + pR_s) * 0.5

            with state_lock:
                st = dict(STATE)

            dst_ctr = ctr + (st["shiftT"]*d)*u + (st["offsetN"]*d)*n
            loc_rgba = get_overlay_for_render()

            fg, a = warp_noshear_rgba(
                loc_rgba, SRC_L, SRC_R, dst_ctr, u, n, d,
                frame.shape, scale=st["scale"], sx=st["scaleX"], sy=st["scaleY"]
            )
            frame = (frame*(1-a) + fg*a).astype(np.uint8)

        # FPS 角标（可删）
        t_now = time.perf_counter()
        dt = t_now - t_prev
        t_prev = t_now
        if dt > 0:
            fps = 1.0/dt
            fps_ema = fps if fps_ema == 0 else 0.9*fps_ema + 0.1*fps
            cv2.putText(frame, f"FPS {fps_ema:4.1f}", (20,40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2, cv2.LINE_AA)

        # JPEG 编码并推流
        ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            continue
        last_jpeg = jpg.tobytes()
        last_frame_ts = time.time()

        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" +
               last_jpeg + b"\r\n")

# ------------ Flask ------------
app = Flask(__name__, static_folder=".", static_url_path="")

@app.route("/")
def root():
    return send_from_directory(".", "index.html")

@app.route("/stream.mjpg")
def stream():
    return Response(generate_mjpeg(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/api/params", methods=["GET", "POST"])
def api_params():
    global overlay_rgba_cached, cached_key
    if request.method == "GET":
        with state_lock:
            return jsonify(STATE)
    data = request.get_json(force=True) or {}
    invalidate = False
    with state_lock:
        for k in ["scale","scaleX","scaleY","shiftT","offsetN","templeRotDeg"]:
            if k in data:
                if k == "templeRotDeg":
                    old = float(STATE.get("templeRotDeg", 0.0))
                    new = float(data[k])
                    if abs(old - new) > 1e-6:
                        invalidate = True
                    STATE[k] = new
                else:
                    STATE[k] = float(data[k])
    if invalidate:
        overlay_rgba_cached = None
        cached_key = (STATE["frameId"], None)
    return jsonify(ok=True, state=STATE)

@app.route("/api/switch_frame", methods=["POST"])
def api_switch_frame():
    global overlay_rgba_base, overlay_rgba_cached, H_ov, W_ov, cached_key, SRC_L, SRC_R
    fid = (request.get_json(force=True) or {}).get("id")
    if fid not in FRAME_FILES:
        return jsonify(ok=False, err="unknown frame id"), 400
    with state_lock:
        STATE["frameId"] = fid
        overlay_rgba_base = load_frame_png(fid)
        overlay_rgba_cached = None
        cached_key = (fid, None)
    # 重新估算锚点（若贴图尺寸不同）
    H_ov, W_ov = overlay_rgba_base.shape[:2]
    SRC_L = (int(0.18*W_ov), int(0.50*H_ov))
    SRC_R = (int(0.82*W_ov), int(0.50*H_ov))
    return jsonify(ok=True, frameId=fid)

@app.route("/snapshot")
def snapshot():
    if last_jpeg is None:
        return "no frame yet", 503
    fn = f"snapshot_{int(last_frame_ts)}.jpg"
    headers = {
        "Content-Type": "image/jpeg",
        "Content-Disposition": f'attachment; filename="{fn}"'
    }
    return Response(last_jpeg, headers=headers)

if __name__ == "__main__":
    # 开发直接跑；生产建议换 waitress/gunicorn
    app.run(host="0.0.0.0", port=8000, threaded=True)

