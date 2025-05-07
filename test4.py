import cv2
import time
import numpy as np


frame_count = 0 
start_time = time.time()

# 创建背景减除器对象
fgbg = cv2.createBackgroundSubtractorMOG2(history=60, varThreshold=200, detectShadows=False)
# fgbg = cv2.createBackgroundSubtractorKNN()
# mog2 = cv2.createBackgroundSubtractorMOG2(history=50, varThreshold=20, detectShadows=False)
# knn = cv2.createBackgroundSubtractorKNN(history=200, dist2Threshold=200.0, detectShadows=False)


# 打开视频文件
# cap = cv2.VideoCapture('sample_9.mp4')
cap = cv2.VideoCapture('./data/5_2_fast1.avi')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    #  获取帧的宽度和高度
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # print(f" w:{frame_width} h:{frame_height}")

    # 定义矩形 ROI 的左上角和右下角坐标
    x, y, width, height = 50, 120, 540, 400

    # 获取 ROI
    roi = frame[y:y + height, x:x + width]
    resized_image = cv2.resize(roi, None, fx=0.2, fy=0.2, interpolation=cv2.INTER_AREA)


    frame_count += 1
    if frame_count % 10 == 0:
        end_time = time.time()
        elapsed_time = end_time - start_time
        fps = frame_count / elapsed_time
        # print(f"当前帧率: {fps:.2f} FPS")

    # 应用背景减除器
    fgmask = fgbg.apply(resized_image)
    sum = np.count_nonzero(fgmask)
    if sum > 5:
        print(f"sum:{sum}")
    # print(fgmask)
    

    # 显示原始帧和前景掩码
    cv2.imshow('Original Frame', resized_image)
    cv2.imshow('Foreground Mask', fgmask)
    

    # 按 'q' 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()
    