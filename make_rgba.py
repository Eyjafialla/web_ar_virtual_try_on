# make_rgba.py
import cv2, numpy as np
img = cv2.imread("Frame_A.png", cv2.IMREAD_COLOR)   # 你的第一张图
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
H,S,V = cv2.split(hsv)
near_white = ((V>230) & (S<20)).astype(np.uint8)*255      # 白底
alpha = cv2.bitwise_not(near_white)                       # 反转=前景
rgba  = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
rgba[:,:,3] = cv2.GaussianBlur(alpha,(3,3),0)             # 轻羽化
cv2.imwrite("Frame_A.png", rgba)
