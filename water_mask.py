import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt



a1=cv.imread("C:/Users/w00508256/Pictures/001854_test.png")
a2=cv.imread("C:/Users/w00508256/Pictures/test.png")

gray = cv2.cvtColor(a2, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY_INV)[1]
backtorgb = cv2.cvtColor(thresh,cv2.COLOR_GRAY2RGB)
(h, w) = backtorgb.shape[:2]
water_mask = np.dstack([backtorgb, np.ones((h, w), dtype="uint8") * 255])
(B, G, R, A) = cv2.split(water_mask)
B = cv2.bitwise_and(B, B, mask=A)
G = cv2.bitwise_and(G, G, mask=A)
R = cv2.bitwise_and(R, R, mask=A)
watermark = cv2.merge([B, G, R, A])

(h, w) = a1.shape[:2]
a1 = np.dstack([a1, np.ones((h, w), dtype="uint8") * 255])

overlay = np.zeros((h, w, 4), dtype="uint8")
(H, W) = water_mask.shape[:2]
watermark_2=cv.resize(watermark,(int(W/2),int(H/2)),interpolation=cv2.INTER_LINEAR)
(H, W) = watermark_2.shape[:2]
overlay[200:H+200,300:W+300] = watermark_2


output = a1.copy()
# alpha 为第一张图片的透明度
alpha =0.25
# beta 为第二张图片的透明度
beta = 1
gamma = 0
output2=cv2.addWeighted(overlay, alpha, output, beta, gamma)
cv2.imwrite("C:/Users/w00508256/Pictures/wm_thresh2.png", output2)
