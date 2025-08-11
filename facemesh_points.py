import cv2
import mediapipe as mp
import math

# 你的机器上 index=1 正常；Windows 后端用 dshow（不稳就换 MSMF）
INDEX = 1
BACKEND = cv2.CAP_DSHOW  # 可改为 cv2.CAP_MSMF

# 关键点ID
LM_LEFT_OUTER  = 263
LM_RIGHT_OUTER = 33
LM_NOSE_BRIDGE = 168

# 打开摄像头
cap = cv2.VideoCapture(INDEX, BACKEND)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
if not cap.isOpened():
    raise RuntimeError("摄像头打开失败，试试换 BACKEND 或把 INDEX 改为 0/1 验证。")

# MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # 可选：镜像成“自拍视图”，更直观
        frame = cv2.flip(frame, 1)

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        res = face_mesh.process(rgb)
        rgb.flags.writeable = True

        if res.multi_face_landmarks:
            face = res.multi_face_landmarks[0].landmark

            def px(idx):
                p = face[idx]
                return int(p.x * w), int(p.y * h)

            pL = px(LM_LEFT_OUTER)
            pR = px(LM_RIGHT_OUTER)
            pN = px(LM_NOSE_BRIDGE)

            # 画点（圆）
            cv2.circle(frame, pL, 4, (0, 255, 0), -1)
            cv2.circle(frame, pR, 4, (0, 255, 0), -1)
            cv2.circle(frame, pN, 4, (0, 255, 0), -1)

            # 标注编号/名称
            cv2.putText(frame, "263", (pL[0]+6, pL[1]-6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2, cv2.LINE_AA)
            cv2.putText(frame, "33",  (pR[0]+6, pR[1]-6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2, cv2.LINE_AA)
            cv2.putText(frame, "168", (pN[0]+6, pN[1]-6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2, cv2.LINE_AA)

            # 画“眼睛基线”与鼻梁辅助线
            cv2.line(frame, pL, pR, (255, 255, 255), 2)
            mid = ((pL[0]+pR[0])//2, (pL[1]+pR[1])//2)
            cv2.line(frame, mid, pN, (200, 200, 200), 1)

            # 可选：算个水平角度，观察头部旋转
            angle = math.degrees(math.atan2(pR[1]-pL[1], pR[0]-pL[0]))
            cv2.putText(frame, f"eye-line angle: {angle:.1f} deg", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)

        cv2.imshow("FaceMesh keypoints (33, 168, 263)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):
            break

cap.release()
cv2.destroyAllWindows()
