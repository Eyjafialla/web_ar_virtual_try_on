import os, time, math, threading
import cv2, numpy as np
from flask import Flask, Response, request, jsonify, send_from_directory
import mediapipe as mp

HERE = os.path.dirname(__file__)
FRAME_DIR = os.path.join(HERE, "frames")

FRAME_IDS = ("A", "B", "C")
DEFAULT_FRAME_ID = "A"

# ---------- Auto-Fit 分档阈值 ----------
# r = d/W（外眼角像素距离/画面宽度），粗准即可
T_SM = float(os.environ.get("AUTO_FIT_T_SM", "0.24"))  # r < T_SM -> S
T_ML = float(os.environ.get("AUTO_FIT_T_ML", "0.28"))  # T_SM <= r < T_ML -> M，否则 L

# 不同规格的相对倍率（体现不同尺码的相对观感；非 UI 可调）
SIZE_SCALE = {
    "S": float(os.environ.get("SIZE_K_S", "0.93")),
    "M": float(os.environ.get("SIZE_K_M", "1.00")),
    "L": float(os.environ.get("SIZE_K_L", "1.07")),
}

# —— 粗调参数（仅内部使用） —— #
ARM_PIVOT_EDGE = os.environ.get("ARM_PIVOT_EDGE", "auto").lower()
EYE2TEMPLE_U   = float(os.environ.get("EYE2TEMPLE_U", "0.14"))
EYE2TEMPLE_N   = float(os.environ.get("EYE2TEMPLE_N", "-0.02"))
W_CAP          = int(os.environ.get("CAP_W", "960"))
H_CAP          = int(os.environ.get("CAP_H", "540"))
CAM_INDEX      = int(os.environ.get("CAM_INDEX", "1"))
SMOOTH_A       = float(os.environ.get("SMOOTH_A", "0.85"))

SIZES = ("S", "M", "L")
DEFAULT_SIZE = "M"

def default_state(fid=DEFAULT_FRAME_ID, sz=DEFAULT_SIZE):
    return {
        "frameId": fid,
        "size": sz,                     # 当前规格
        "fitMode": "once",              # 'once' | 'manual'
        "fitDone": False,               # once模式是否已完成匹配
        # Global（内部使用；UI 不暴露）
        "scale": SIZE_SCALE.get(sz, 1), "scaleX":1.0, "scaleY":1.0,
        "shiftT":0.0, "offsetN":0.0,
        "templeRotDeg": 5.0,
        # 镜臂微调（UI 可调）
        "templeScaleL":1.0, "templeShiftTL":0.0, "templeOffsetNL":0.0, "templeRotL": 5.0,
        "templeScaleR":1.0, "templeShiftTR":0.0, "templeOffsetNR":0.0, "templeRotR": -5.0,
        "pivotMode":"eye_outer",
        "autoHideTemples": False,
    }

state_lock = threading.Lock()
STATE = default_state()

# ---------- 读图 ----------
def _read_rgba(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Missing image: {path}")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    elif img.shape[2] == 3:
        b,g,r = cv2.split(img); a = np.full(b.shape, 255, np.uint8)
        img = cv2.merge([b,g,r,a])
    return img

def pivot_from_alpha(img_rgba, side='L'):
    h, w = img_rgba.shape[:2]
    a = img_rgba[...,3] > 0
    ys, xs = np.where(a)
    if xs.size == 0:
        x = int((0.18 if side=='L' else 0.82) * w); y = int(0.5*h)
        eps = max(1, int(0.003*w))
        return ((x, y), (x+eps, y)) if side=='L' else ((x-eps, y), (x, y))

    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    y = int((y0 + y1)//2)
    eps = max(1, int(0.003*w))

    if side=='L':
        inner  = ((x1, y), (x1+eps, y))
        outer  = ((x0, y), (x0+eps, y))
        if   ARM_PIVOT_EDGE == 'inner': pick = inner
        elif ARM_PIVOT_EDGE == 'outer': pick = outer
        else: pick = outer if x0 < x1 else inner
    else:
        inner  = ((x0-eps, y), (x0, y))
        outer  = ((x1-eps, y), (x1, y))
        if   ARM_PIVOT_EDGE == 'inner': pick = inner
        elif ARM_PIVOT_EDGE == 'outer': pick = outer
        else: pick = outer if x1 > x0 else inner
    return pick

def load_layers(fid, size_label):
    def p(part):
        sized = os.path.join(FRAME_DIR, f"Frame_{fid}_{part}_{size_label}.png")
        plain = os.path.join(FRAME_DIR, f"Frame_{fid}_{part}.png")
        return sized if os.path.exists(sized) else plain
    front = p("front")
    left  = p("left_arm")
    right = p("right_arm")
    L = {"front": _read_rgba(front)}
    if os.path.exists(left):  L["left"]  = _read_rgba(left)
    if os.path.exists(right): L["right"] = _read_rgba(right)
    return L

# 初始化图层（默认 M）
layers = load_layers(DEFAULT_FRAME_ID, DEFAULT_SIZE)
H_ov, W_ov = layers["front"].shape[:2]
SRC_L = (int(0.18*W_ov), int(0.50*H_ov))
SRC_R = (int(0.82*W_ov), int(0.50*H_ov))
SRC_L_PIVOT = pivot_from_alpha(layers.get("left"),  'L') if "left"  in layers else ((SRC_L, (SRC_L[0]+1, SRC_L[1])))
SRC_R_PIVOT = pivot_from_alpha(layers.get("right"), 'R') if "right" in layers else (((SRC_R[0]-1, SRC_R[1]), SRC_R))
FRONT_BASE_DIST = float(np.linalg.norm(np.array(SRC_R) - np.array(SRC_L))) + 1e-6

# ---------- MediaPipe ----------
mp_fm = mp.solutions.face_mesh
fm = mp_fm.FaceMesh(max_num_faces=1, refine_landmarks=False,
                    min_detection_confidence=0.5, min_tracking_confidence=0.5)
def to_px(lm, w, h): return np.array([lm.x*w, lm.y*h], np.float32)

# ---------- 仿射 ----------
def warp_rgba(rgba, src_L, src_R, dst_ctr, u, n, d, frame_shape,
              scale=1.0, sx=1.0, sy=1.0, theta_add_deg=0.0, base_dist_ref=None):
    Hf,Wf = frame_shape[:2]
    ovL, ovR = np.array(src_L, np.float32), np.array(src_R, np.float32)
    ov_ctr  = (ovL + ovR) * 0.5
    ov_dist = float(np.linalg.norm(ovR - ovL)) + 1e-6
    ref = float(base_dist_ref) if base_dist_ref is not None else ov_dist
    sx = (d/ref) * float(scale) * float(sx)
    sy = (d/ref) * float(scale) * float(sy)
    theta = math.atan2(float(u[1]), float(u[0])) + math.radians(float(theta_add_deg))
    c,s = math.cos(theta), math.sin(theta)
    M = np.array([[ sx*c, -sy*s, 0.0 ],
                  [ sx*s,  sy*c, 0.0 ]], np.float32)
    M[:,2] = dst_ctr - (M[:,:2] @ ov_ctr)
    interp = cv2.INTER_LINEAR if max(sx,sy) > 0.98 else cv2.INTER_AREA
    warped = cv2.warpAffine(rgba, M, (Wf,Hf), flags=interp, borderMode=cv2.BORDER_TRANSPARENT)
    b,g,r,a = cv2.split(warped)
    return cv2.merge([b,g,r]).astype(np.float32), (a.astype(np.float32)/255.0)[...,None]

# ---------- 采集 ----------
cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  W_CAP)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H_CAP)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

last_jpeg, last_ts = None, 0

# EMA
smooth = {"pL":None,"pR":None,"u":None,"ctr":None,"d":None}
def ema(prev, new, a): return new if prev is None else (a*prev + (1.0-a)*new)

# ---------- 尺寸切换与一次性 Auto-Fit ----------
def _apply_size_and_reload(size_label):
    """切换规格：设置内部倍率，并按当前 frameId 重载对应尺寸贴图"""
    global layers, SRC_L, SRC_R, SRC_L_PIVOT, SRC_R_PIVOT, H_ov, W_ov, FRONT_BASE_DIST
    with state_lock:
        STATE["size"]  = size_label
        STATE["scale"] = SIZE_SCALE.get(size_label, 1.0)
        fid = STATE.get("frameId", DEFAULT_FRAME_ID)
    layers = load_layers(fid, size_label)
    H_ov, W_ov = layers["front"].shape[:2]
    SRC_L = (int(0.18*W_ov), int(0.50*H_ov))
    SRC_R = (int(0.82*W_ov), int(0.50*H_ov))
    SRC_L_PIVOT = pivot_from_alpha(layers.get("left"),  'L') if "left"  in layers else ((SRC_L, (SRC_L[0]+1, SRC_L[1])))
    SRC_R_PIVOT = pivot_from_alpha(layers.get("right"), 'R') if "right" in layers else (((SRC_R[0]-1, SRC_R[1]), SRC_R))
    FRONT_BASE_DIST = float(np.linalg.norm(np.array(SRC_R) - np.array(SRC_L))) + 1e-6

def _autofit_size(d_px, W_img):
    """根据 r=d/W 选择 S/M/L"""
    if W_img <= 0: return "M"
    r = float(d_px) / float(W_img)
    if   r < T_SM: return "S"
    elif r < T_ML: return "M"
    else:          return "L"

# ---------- 推流 ----------
def generate_stream(jpg_quality=80):
    global last_jpeg, last_ts
    fps_ema, t_prev = 0.0, time.perf_counter()

    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.01); continue
        frame = cv2.flip(frame, 1)
        H, W = frame.shape[:2]

        with state_lock: st = dict(STATE)

        res = fm.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if res.multi_face_landmarks:
            lm = res.multi_face_landmarks[0].landmark

            pL_eye = to_px(lm[263], W, H)
            pR_eye = to_px(lm[33],  W, H)

            # 保证 pL 在左、pR 在右
            if pL_eye[0] <= pR_eye[0]:
                p_imgL, p_imgR = pL_eye, pR_eye
            else:
                p_imgL, p_imgR = pR_eye, pL_eye
            v_dir = p_imgR - p_imgL
            d_raw = float(np.linalg.norm(v_dir)) + 1e-6
            u_raw = v_dir / (np.linalg.norm(v_dir) + 1e-6)
            n_raw = np.array([-u_raw[1], u_raw[0]], np.float32)
            if n_raw[1] < 0: n_raw = -n_raw
            ctr_raw = (pL_eye + pR_eye) * 0.5

            # 平滑
            smooth["pL"] = ema(smooth["pL"], pL_eye, SMOOTH_A)
            smooth["pR"] = ema(smooth["pR"], pR_eye, SMOOTH_A)
            smooth["u"]  = ema(smooth["u"],  u_raw,  SMOOTH_A)
            if smooth["u"] is not None:
                nrm = np.linalg.norm(smooth["u"])
                if nrm > 1e-6: smooth["u"] /= nrm
            smooth["ctr"]= ema(smooth["ctr"], ctr_raw, SMOOTH_A)
            smooth["d"]  = float(ema(smooth["d"], d_raw, SMOOTH_A))

            pL_eye, pR_eye = smooth["pL"], smooth["pR"]
            u_dir, d_scale = smooth["u"], smooth["d"]
            n_dir = np.array([-u_dir[1], u_dir[0]], np.float32)
            if n_dir[1] < 0: n_dir = -n_dir
            center_eyes = (pL_eye + pR_eye) * 0.5

            # === 一次性 Auto-Fit（仅在 fitMode=once 且尚未 fitDone 时触发） ===
            with state_lock:
                fitMode = STATE.get("fitMode","once")
                fitDone = bool(STATE.get("fitDone", False))
            if fitMode == "once" and not fitDone:
                cand = _autofit_size(d_scale, W)
                _apply_size_and_reload(cand)
                with state_lock:
                    STATE["fitDone"] = True  # 锁定；之后不再改动

            # 取渲染参数
            with state_lock:
                gT = float(STATE["shiftT"]); gN = float(STATE["offsetN"])
                gScale = float(STATE["scale"])
                gScaleX = float(STATE["scaleX"]); gScaleY = float(STATE["scaleY"])
                gRot    = float(STATE["templeRotDeg"])
                autoHide = bool(STATE.get("autoHideTemples", False))
                sz_label = STATE.get("size","?")

            effX = gScale * gScaleX
            effY = gScale * gScaleY

            # 锚点
            u_off = EYE2TEMPLE_U * d_scale * effX
            n_off = EYE2TEMPLE_N * d_scale * effY
            pL_anchor = pL_eye + (+u_off)*u_dir + (n_off)*n_dir
            pR_anchor = pR_eye + (-u_off)*u_dir + (n_off)*n_dir

            # auto-hide（基于左右眼 z 深度的简易判断）
            hideL = hideR = False
            if autoHide:
                zL, zR = lm[263].z, lm[33].z
                dz, TH = float(zR - zL), 0.003
                if   abs(dz) <= TH: hideL = hideR = False
                elif dz > TH:      hideL, hideR = False, True
                else:              hideL, hideR = True,  False

            # 镜臂（可调）
            with state_lock:
                stL = (STATE["templeScaleL"], STATE["templeShiftTL"], STATE["templeOffsetNL"], STATE["templeRotL"])
                stR = (STATE["templeScaleR"], STATE["templeShiftTR"], STATE["templeOffsetNR"], STATE["templeRotR"])

            if "left" in layers and not hideL:
                fg, a = warp_rgba(
                    layers["left"], SRC_L_PIVOT[0], SRC_L_PIVOT[1],
                    pL_anchor + (stL[1]*d_scale*effX)*u_dir + (stL[2]*d_scale*effY)*n_dir,
                    u_dir, n_dir, d_scale, frame.shape,
                    scale=stL[0]*gScale, sx=gScaleX, sy=gScaleY,
                    theta_add_deg=float(STATE["templeRotDeg"])+float(stL[3]),
                    base_dist_ref=FRONT_BASE_DIST
                )
                frame = (frame*(1-a) + fg*a).astype(np.uint8)

            if "right" in layers and not hideR:
                fg, a = warp_rgba(
                    layers["right"], SRC_R_PIVOT[0], SRC_R_PIVOT[1],
                    pR_anchor + (stR[1]*d_scale*effX)*u_dir + (stR[2]*d_scale*effY)*n_dir,
                    u_dir, n_dir, d_scale, frame.shape,
                    scale=stR[0]*gScale, sx=gScaleX, sy=gScaleY,
                    theta_add_deg=-float(STATE["templeRotDeg"])+float(stR[3]),
                    base_dist_ref=FRONT_BASE_DIST
                )
                frame = (frame*(1-a) + fg*a).astype(np.uint8)

            # 前框
            dst_ctr = center_eyes + (gT*d_scale)*u_dir + (gN*d_scale)*n_dir
            fg, a = warp_rgba(
                layers["front"], SRC_L, SRC_R, dst_ctr,
                u_dir, n_dir, d_scale, frame.shape,
                scale=gScale, sx=gScaleX, sy=gScaleY
            )
            frame = (frame*(1-a) + fg*a).astype(np.uint8)

            # HUD：当前 size
            cv2.putText(frame, f"Size:{sz_label}", (14, 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,255), 2, cv2.LINE_AA)

        # FPS
        t = time.perf_counter(); dt = t - t_prev; t_prev = t
        if dt > 0:
            fps = 1.0/dt; fps_ema = fps if fps_ema==0 else 0.9*fps_ema + 0.1*fps
            cv2.putText(frame, f"{fps_ema:4.1f} FPS", (W-200, 42),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2, cv2.LINE_AA)

        ok, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, jpg_quality])
        if not ok: continue
        last_jpeg, last_ts = buf.tobytes(), time.time()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + last_jpeg + b'\r\n')

# ---------- Flask ----------
app = Flask(__name__, static_folder=".", static_url_path="")

@app.route("/")
def root():
    # 前端文件名沿用 index1.html
    return send_from_directory(".", "index.html")

@app.route("/stream.mjpg")
def stream_jpg():
    return Response(generate_stream(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/snapshot")
def snapshot():
    if last_jpeg is None: return "no frame yet", 503
    return Response(last_jpeg, headers={
        "Content-Type":"image/jpeg",
        "Content-Disposition": f'attachment; filename="snapshot_{int(last_ts)}.jpg"'
    })

@app.route("/api/params", methods=["GET","POST"])
def api_params():
    if request.method == "GET":
        with state_lock: return jsonify(STATE)
    data = request.get_json(force=True) or {}
    # 仅接受：镜臂微调 + autoHide（无 global/occlusion）
    with state_lock:
        for k in ["templeScaleL","templeShiftTL","templeOffsetNL","templeRotL",
                  "templeScaleR","templeShiftTR","templeOffsetNR","templeRotR",
                  "autoHideTemples"]:
            if k in data:
                STATE[k] = bool(data[k]) if k=="autoHideTemples" else float(data[k])
        STATE["pivotMode"] = "eye_outer"
    return jsonify(ok=True, state=STATE)

@app.route("/api/size", methods=["POST"])
def api_size():
    """手动切换 S/M/L，并锁定为手动模式"""
    data = request.get_json(force=True) or {}
    sz = (data.get("size") or "").upper()
    if sz not in SIZES:
        return jsonify(ok=False, err="size must be S/M/L"), 400
    with state_lock:
        STATE["fitMode"] = "manual"
        STATE["fitDone"] = True
    _apply_size_and_reload(sz)
    with state_lock:
        cur = dict(STATE)
    return jsonify(ok=True, state=cur)

@app.route("/api/fit_once", methods=["POST"])
def api_fit_once():
    """复位为“一次自动匹配”模式；下一帧检测到人脸即会匹配一次并锁定"""
    with state_lock:
        STATE["fitMode"] = "once"
        STATE["fitDone"] = False
    return jsonify(ok=True, state=STATE)

def _switch_frame(fid):
    with state_lock:
        STATE["frameId"] = fid
        size_label = STATE.get("size", DEFAULT_SIZE)
    _apply_size_and_reload(size_label)

@app.route("/api/switch_frame", methods=["POST"])
def api_switch_frame():
    data = request.get_json(force=True) or {}
    fid = data.get("id") or data.get("frame")
    if fid not in FRAME_IDS:
        return jsonify(ok=False, err="unknown frame id"), 400
    _switch_frame(fid)
    return jsonify(ok=True, frameId=fid)

@app.route("/api/reset", methods=["POST"])
def api_reset():
    with state_lock:
        fid = STATE.get("frameId", DEFAULT_FRAME_ID)
        STATE.clear(); STATE.update(default_state(fid, DEFAULT_SIZE))
    _apply_size_and_reload(DEFAULT_SIZE)
    return jsonify(ok=True, state=STATE)

@app.route("/api/debug_layers")
def api_debug_layers():
    info = {}
    for k, img in layers.items():
        h,w,c = img.shape
        a = cv2.split(img)[3] if c==4 else np.full((h,w),255,np.uint8)
        info[k] = {"shape":[h,w,int(c)], "alpha_ratio": float((a>0).mean())}
    return jsonify(info)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, threaded=True)
