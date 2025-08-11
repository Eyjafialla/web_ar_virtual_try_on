import argparse
import time
import platform
import cv2

# ---------- 后端选择 ----------
def default_backends():
    sysname = platform.system().lower()
    if sysname == "windows":
        return [cv2.CAP_DSHOW, cv2.CAP_MSMF]  # Windows 常用
    elif sysname == "darwin":
        return [cv2.CAP_AVFOUNDATION]         # macOS
    else:
        return [cv2.CAP_V4L2]                 # Linux

BACKEND_NAME2CONST = {
    "auto": None,
    "dshow": cv2.CAP_DSHOW,
    "msmf": cv2.CAP_MSMF,
    "avfoundation": cv2.CAP_AVFOUNDATION,
    "v4l2": cv2.CAP_V4L2,
}

def backend_const_to_name(b):
    for k, v in BACKEND_NAME2CONST.items():
        if v == b:
            return k
    return str(b)

# ---------- 探测 ----------
def try_open_once(index, backend):
    if backend is None:
        cap = cv2.VideoCapture(index)
    else:
        cap = cv2.VideoCapture(index, backend)
    ok_opened = cap.isOpened()
    ok_frame, frame = cap.read() if ok_opened else (False, None)
    if not ok_frame and ok_opened:
        # 再读一次，某些后端第一次 read 可能为空
        ok_frame, frame = cap.read()
    cap.release()
    shape = None if frame is None else frame.shape
    return ok_opened, ok_frame, shape

def probe_cameras(max_index=5, backends=None):
    results = []
    if backends is None:
        backends = default_backends()
    for b in backends:
        for i in range(max_index + 1):
            ok_open, ok_fr, shape = try_open_once(i, b)
            results.append({
                "index": i,
                "backend": b,
                "opened": ok_open,
                "frame_ok": ok_fr,
                "shape": shape
            })
    return results

# ---------- 打开摄像头 ----------
def open_camera(index=None, backend=None, max_index=5):
    # 指定 index：直接尝试
    if index is not None:
        cands = [backend] if backend is not None else default_backends()
        for b in cands:
            cap = cv2.VideoCapture(index, b) if b is not None else cv2.VideoCapture(index)
            if cap.isOpened():
                ok, _ = cap.read()
                if ok:
                    return cap, (index, b)
            cap.release()
        return None, (index, backend)

    # 未指定 index：自动探测
    backs = [backend] if backend is not None else default_backends()
    for b in backs:
        for i in range(max_index + 1):
            cap = cv2.VideoCapture(i, b) if b is not None else cv2.VideoCapture(i)
            if cap.isOpened():
                ok, _ = cap.read()
                if ok:
                    return cap, (i, b)
            cap.release()
    return None, (None, backend)

# ---------- 主程序 ----------
def main():
    ap = argparse.ArgumentParser(description="Camera + MediaPipe FaceMesh test")
    ap.add_argument("--list", action="store_true", help="列出可用摄像头后退出")
    ap.add_argument("--index", type=int, default=None, help="指定摄像头编号")
    ap.add_argument("--backend", type=str, default="auto",
                    choices=list(BACKEND_NAME2CONST.keys()),
                    help="指定后端：auto/dshow/msmf/avfoundation/v4l2")
    ap.add_argument("--max-index", type=int, default=5, help="探测的最大编号（含）")
    ap.add_argument("--width", type=int, default=1280, help="请求的宽")
    ap.add_argument("--height", type=int, default=720, help="请求的高")
    ap.add_argument("--no-mp", action="store_true", help="不启用 MediaPipe FaceMesh")
    args = ap.parse_args()

    backend_const = BACKEND_NAME2CONST[args.backend]

    if args.list:
        rows = probe_cameras(max_index=args.max_index,
                             backends=[backend_const] if backend_const is not None else default_backends())
        print("== 探测结果 ==")
        for r in rows:
            bname = backend_const_to_name(r["backend"])
            print(f"backend={bname:12s} index={r['index']} opened={r['opened']} frame_ok={r['frame_ok']} shape={r['shape']}")
        return

    # 打开摄像头
    cap, chosen = open_camera(index=args.index, backend=backend_const, max_index=args.max_index)
    if cap is None:
        bname = backend_const_to_name(backend_const)
        print(f"[错误] 无法打开摄像头（index={args.index}, backend={bname}）。"
              f"尝试 --list 查看可用设备，或关闭占用相机的应用（如 Zoom/Teams/浏览器等）。")
        return

    idx, bused = chosen
    bname = backend_const_to_name(bused)
    print(f"[OK] 已打开摄像头 index={idx}, backend={bname}")

    # 设置分辨率
    if args.width and args.height:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    # MediaPipe FaceMesh
    use_mp = not args.no_mp
    face_mesh = None
    mp_draw = mp_styles = None
    if use_mp:
        try:
            import mediapipe as mp
            mp_face_mesh = mp.solutions.face_mesh
            mp_draw = mp.solutions.drawing_utils
            mp_styles = mp.solutions.drawing_styles
            face_mesh = mp_face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            print("[OK] MediaPipe FaceMesh 已启用")
        except Exception as e:
            print(f"[警告] 初始化 MediaPipe 失败，将仅显示相机画面：{e}")
            use_mp = False

    # 采集循环
    last = time.time()
    fps = 0.0
    win_title = f"Camera Test (index={idx}, backend={bname}) - FaceMesh: {'ON' if use_mp else 'OFF'}"
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            print("[警告] 读取帧失败，可能被占用或设备掉线。")
            break

        # FPS 估算
        now = time.time()
        dt = now - last
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt)
        last = now

        if use_mp and face_mesh is not None:
            # BGR -> RGB（MP 需要 RGB）
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = face_mesh.process(rgb)
            rgb.flags.writeable = True

            if results.multi_face_landmarks:
                for lm in results.multi_face_landmarks:
                    # 三种可视化叠加：网格、轮廓、虹膜
                    mp_draw.draw_landmarks(
                        frame, lm,
                        mp.solutions.face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_styles.get_default_face_mesh_tesselation_style()
                    )
                    mp_draw.draw_landmarks(
                        frame, lm,
                        mp.solutions.face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_styles.get_default_face_mesh_contours_style()
                    )
                    mp_draw.draw_landmarks(
                        frame, lm,
                        mp.solutions.face_mesh.FACEMESH_IRISES,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_styles.get_default_face_mesh_iris_connections_style()
                    )

        # 叠字：分辨率 & FPS
        h, w = frame.shape[:2]
        text = f"{w}x{h} | FPS: {fps:.1f}"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow(win_title, frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):   # ESC or 'q'
            break
        if key == ord('s'):
            fname = f"snapshot_{int(time.time())}.png"
            cv2.imwrite(fname, frame)
            print(f"[保存] {fname}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
