# -*- coding: utf-8 -*-
"""
Web AR Glasses (A/B/C; front/left_arm/right_arm)
- 只用外眼角 263/33 作为绑定锚点
- Global(Scale/Width/Height/Shift/Offset/Temple rot) 统一作用：前框 + 两侧镜臂 + 锚点 + 镜臂微调
- 锚点偏移与镜臂微调随 Global 等比（effX=scale*scaleX, effY=scale*scaleY）
- 铰链取点自适应(inner/outer/auto)，水平外移随 Scale 轻量自适应
- 可选遮挡（默认关），MJPEG：/stream.mjpg
"""
import os, time, math, threading
import cv2, numpy as np
from flask import Flask, Response, request, jsonify, send_from_directory
import mediapipe as mp

HERE = os.path.dirname(__file__)
FRAME_DIR = os.path.join(HERE, "frames")
FRAME_IDS = ("A", "B", "C")
DEFAULT_FRAME_ID = "A"

# —— 粗调参数（可用环境变量覆盖） ——
# 'auto'：自动挑更靠耳侧（outer）的铰链；也可设 'inner' 或 'outer'
ARM_PIVOT_EDGE = os.environ.get("ARM_PIVOT_EDGE", "auto").lower()
EYE2TEMPLE_U   = float(os.environ.get("EYE2TEMPLE_U", "0.14"))   # 沿眼线外移比例（相对眼距）
EYE2TEMPLE_N   = float(os.environ.get("EYE2TEMPLE_N", "-0.02"))  # 法向微调比例
W_CAP          = int(os.environ.get("CAP_W", "960"))
H_CAP          = int(os.environ.get("CAP_H", "540"))
CAM_INDEX      = int(os.environ.get("CAM_INDEX", "1"))
SMOOTH_A       = float(os.environ.get("SMOOTH_A", "0.85"))       # 指数平滑

def default_state(fid=DEFAULT_FRAME_ID):
    return {
        "frameId": fid,
        # Global
        "scale":1.0, "scaleX":1.0, "scaleY":1.0,
        "shiftT":0.0, "offsetN":0.0,
        "templeRotDeg": 5.0,  # 左臂 +gRot，右臂 -gRot
        # 镜臂微调（屏幕右→后端L；屏幕左→后端R）
        "templeScaleL":1.0, "templeShiftTL":0.0, "templeOffsetNL":0.0, "templeRotL": 5.0,
        "templeScaleR":1.0, "templeShiftTR":0.0, "templeOffsetNR":0.0, "templeRotR": -5.0,
        # 固定外眼角
        "pivotMode":"eye_outer",
        # 可选
        "occlusion": False,
        "autoHideTemples": False,
    }

state_lock = threading.Lock()
STATE = default_state()

# ---------- 贴图 ----------
def _read_rgba(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None: raise FileNotFoundError(f"Missing image: {path}")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    elif img.shape[2] == 3:
        b,g,r = cv2.split(img); a = np.full(b.shape, 255, np.uint8)
        img = cv2.merge([b,g,r,a])
    return img

def pivot_from_alpha(img_rgba, side='L'):
    """
    从镜臂 PNG 的 alpha 取近铰链的短线（局部两点）。
    支持 inner/outer/auto：auto 会自动选更靠耳侧（outer）的候选。
    """
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

    # 两个候选：inner/outer
    if side=='L':
        inner  = ((x1, y), (x1+eps, y))        # 靠镜框
        outer  = ((x0, y), (x0+eps, y))        # 靠耳朵
        if ARM_PIVOT_EDGE == 'inner': pick = inner
        elif ARM_PIVOT_EDGE == 'outer': pick = outer
        else:  # auto：选更靠耳侧（x 更小）
            pick = outer if x0 < x1 else inner
    else:
        inner  = ((x0-eps, y), (x0, y))
        outer  = ((x1-eps, y), (x1, y))
        if ARM_PIVOT_EDGE == 'inner': pick = inner
        elif ARM_PIVOT_EDGE == 'outer': pick = outer
        else:  # auto：选更靠耳侧（x 更大）
            pick = outer if x1 > x0 else inner
    return pick

def load_layers(fid):
    front = os.path.join(FRAME_DIR, f"Frame_{fid}_front.png")
    left  = os.path.join(FRAME_DIR, f"Frame_{fid}_left_arm.png")
    right = os.path.join(FRAME_DIR, f"Frame_{fid}_right_arm.png")
    L = {"front": _read_rgba(front)}
    if os.path.exists(left):  L["left"]  = _read_rgba(left)
    if os.path.exists(right): L["right"] = _read_rgba(right)
    return L

layers = load_layers(DEFAULT_FRAME_ID)
H_ov, W_ov = layers["front"].shape[:2]
SRC_L = (int(0.18*W_ov), int(0.50*H_ov))
SRC_R = (int(0.82*W_ov), int(0.50*H_ov))
SRC_L_PIVOT = pivot_from_alpha(layers["left"],  'L') if "left"  in layers else ((SRC_L, (SRC_L[0]+1, SRC_L[1])))
SRC_R_PIVOT = pivot_from_alpha(layers["right"], 'R') if "right" in layers else (((SRC_R[0]-1, SRC_R[1]), SRC_R))
FRONT_BASE_DIST = float(np.linalg.norm(np.array(SRC_R) - np.array(SRC_L))) + 1e-6

# ---------- MediaPipe ----------
mp_fm = mp.solutions.face_mesh
fm = mp_fm.FaceMesh(max_num_faces=1, refine_landmarks=False,
                    min_detection_confidence=0.5, min_tracking_confidence=0.5)
LM_LEFT_OUTER, LM_RIGHT_OUTER = 263, 33  # 解剖学左/右（人物视角）
def to_px(lm, w, h): return np.array([lm.x*w, lm.y*h], np.float32)

# ---------- 遮挡（可选） ----------
mp_seg = mp.solutions.selfie_segmentation
segmenter = mp_seg.SelfieSegmentation(model_selection=1)
def get_face_mask(frame_bgr):
    H,W = frame_bgr.shape[:2]
    small = cv2.resize(frame_bgr, (256,256), interpolation=cv2.INTER_AREA)
    res = segmenter.process(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
    m = (res.segmentation_mask*255).astype(np.uint8)
    m = cv2.resize(m, (W,H), interpolation=cv2.INTER_LINEAR)
    m = cv2.erode(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)), 1)
    return (m.astype(np.float32)/255.0)[...,None]

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

# ---------- 流 ----------
def generate_stream(jpg_quality=80):
    global last_jpeg, last_ts, layers, SRC_L, SRC_R, SRC_L_PIVOT, SRC_R_PIVOT, FRONT_BASE_DIST
    fps_ema, t_prev = 0.0, time.perf_counter()
    face_mask_cached, frame_id = None, 0

    while True:
        ok, frame = cap.read()
        if not ok: time.sleep(0.01); continue
        frame = cv2.flip(frame, 1)
        H, W = frame.shape[:2]

        with state_lock: st = dict(STATE)

        occl = bool(st.get("occlusion", False))
        if occl and ((frame_id % 3) == 0 or face_mask_cached is None):
            face_mask_cached = get_face_mask(frame)

        res = fm.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if res.multi_face_landmarks:
            lm = res.multi_face_landmarks[0].landmark

            # 外眼角（不按 x 交换，保持解剖学左右）
            pL_eye = to_px(lm[LM_LEFT_OUTER],  W, H)   # 人物左=屏幕右
            pR_eye = to_px(lm[LM_RIGHT_OUTER], W, H)   # 人物右=屏幕左

            # 基础方向与距离（屏幕左->屏幕右）
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

            # ===== Global（必须在锚点之前）=====
            gT = float(st["shiftT"]); gN = float(st["offsetN"])
            gScale  = float(st["scale"]);  gScaleX = float(st["scaleX"]); gScaleY = float(st["scaleY"])
            gRot    = float(st["templeRotDeg"])
            effX = gScale * gScaleX   # 横向最终比例
            effY = gScale * gScaleY   # 纵向最终比例

            # —— 锚点（随 Global 等比 + 轻量自适应）——
            alpha = min(0.25, max(0.0, gScale - 1.0) * 0.20)  # S>1 时最多 +25%
            u_off = EYE2TEMPLE_U * d_scale * effX * (1.0 + alpha)
            n_off = EYE2TEMPLE_N * d_scale * effY
            pL_anchor = pL_eye + (+u_off)*u_dir + (n_off)*n_dir  # 后端L=屏幕右=人物左
            pR_anchor = pR_eye + (-u_off)*u_dir + (n_off)*n_dir  # 后端R=屏幕左=人物右

            # —— 自动隐藏（基于外眼角 z 的远近）——
            hideL = hideR = False
            if st.get("autoHideTemples", False):
                zL, zR = lm[LM_LEFT_OUTER].z, lm[LM_RIGHT_OUTER].z
                dz, TH = float(zR - zL), 0.003
                if abs(dz) <= TH:
                    hideL = hideR = False
                elif dz > TH:
                    hideL, hideR = False, True   # 右更远：藏人物右（屏幕左）
                else:
                    hideL, hideR = True,  False  # 左更远：藏人物左（屏幕右）

            # ===== 镜臂（微调也随 Global 比例）=====
            if "left" in layers and not hideL:
                dst_L = pL_anchor \
                      + ((gT + float(st["templeShiftTL"])) * d_scale * effX) * u_dir \
                      + ((gN + float(st["templeOffsetNL"])) * d_scale * effY) * n_dir
                thetaL = gRot + float(st.get("templeRotL", 0.0))
                fg, a = warp_rgba(
                    layers["left"], SRC_L_PIVOT[0], SRC_L_PIVOT[1], dst_L,
                    u_dir, n_dir, d_scale, frame.shape,
                    scale=float(st["templeScaleL"]) * gScale,
                    sx=gScaleX, sy=gScaleY,
                    theta_add_deg=thetaL, base_dist_ref=FRONT_BASE_DIST
                )
                back = a * (1.0 - face_mask_cached) if (occl and face_mask_cached is not None) else a
                frame = (frame*(1-back) + fg*back).astype(np.uint8)

            if "right" in layers and not hideR:
                dst_R = pR_anchor \
                      + ((gT + float(st["templeShiftTR"])) * d_scale * effX) * u_dir \
                      + ((gN + float(st["templeOffsetNR"])) * d_scale * effY) * n_dir
                thetaR = -gRot + float(st.get("templeRotR", 0.0))
                fg, a = warp_rgba(
                    layers["right"], SRC_R_PIVOT[0], SRC_R_PIVOT[1], dst_R,
                    u_dir, n_dir, d_scale, frame.shape,
                    scale=float(st["templeScaleR"]) * gScale,
                    sx=gScaleX, sy=gScaleY,
                    theta_add_deg=thetaR, base_dist_ref=FRONT_BASE_DIST
                )
                back = a * (1.0 - face_mask_cached) if (occl and face_mask_cached is not None) else a
                frame = (frame*(1-back) + fg*back).astype(np.uint8)

            # ===== 前框（同一套比例）=====
            dst_ctr = center_eyes + (gT*d_scale)*u_dir + (gN*d_scale)*n_dir
            fg, a = warp_rgba(
                layers["front"], SRC_L, SRC_R, dst_ctr,
                u_dir, n_dir, d_scale, frame.shape,
                scale=gScale, sx=gScaleX, sy=gScaleY
            )
            frame = (frame*(1-a) + fg*a).astype(np.uint8)

            # —— DBG 叠字 —— 
            dbg = f"S={gScale:.2f} SX={gScaleX:.2f} SY={gScaleY:.2f}"
            cv2.putText(frame, dbg, (14, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,255), 2, cv2.LINE_AA)

        # FPS
        t = time.perf_counter(); dt = t - t_prev; t_prev = t
        if dt > 0:
            fps = 1.0/dt; fps_ema = fps if fps_ema==0 else 0.9*fps_ema + 0.1*fps
            cv2.putText(frame, f"{fps_ema:4.1f} FPS", (W-200, 42),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2, cv2.LINE_AA)

        ok, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok: continue
        last_jpeg, last_ts = buf.tobytes(), time.time()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + last_jpeg + b'\r\n')
        frame_id += 1

# ---------- Flask ----------
app = Flask(__name__, static_folder=".", static_url_path="")

@app.route("/")
def root():
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
    with state_lock:
        for k in ["scale","scaleX","scaleY","shiftT","offsetN","templeRotDeg",
                  "templeScaleL","templeShiftTL","templeOffsetNL","templeRotL",
                  "templeScaleR","templeShiftTR","templeOffsetNR","templeRotR",
                  "occlusion","autoHideTemples"]:
            if k in data:
                if k in ("occlusion","autoHideTemples"):
                    STATE[k] = bool(data[k])
                else:
                    STATE[k] = float(data[k])
        STATE["pivotMode"] = "eye_outer"
    return jsonify(ok=True, state=STATE)

def _switch(fid):
    global layers, SRC_L, SRC_R, SRC_L_PIVOT, SRC_R_PIVOT, H_ov, W_ov, FRONT_BASE_DIST
    layers = load_layers(fid)
    H_ov, W_ov = layers["front"].shape[:2]
    SRC_L = (int(0.18*W_ov), int(0.50*H_ov))
    SRC_R = (int(0.82*W_ov), int(0.50*H_ov))
    SRC_L_PIVOT = pivot_from_alpha(layers.get("left"),  'L') if "left"  in layers else ((SRC_L, (SRC_L[0]+1, SRC_L[1])))
    SRC_R_PIVOT = pivot_from_alpha(layers.get("right"), 'R') if "right" in layers else (((SRC_R[0]-1, SRC_R[1]), SRC_R))
    FRONT_BASE_DIST = float(np.linalg.norm(np.array(SRC_R) - np.array(SRC_L))) + 1e-6

@app.route("/api/switch_frame", methods=["POST"])
def api_switch_frame():
    data = request.get_json(force=True) or {}
    fid = data.get("id") or data.get("frame")
    if fid not in FRAME_IDS: return jsonify(ok=False, err="unknown frame id"), 400
    _switch(fid)
    with state_lock: STATE["frameId"] = fid
    return jsonify(ok=True, frameId=fid)

@app.route("/api/reset", methods=["POST"])
def api_reset():
    with state_lock:
        fid = STATE.get("frameId", DEFAULT_FRAME_ID)
        STATE.clear(); STATE.update(default_state(fid))
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
