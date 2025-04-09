import cv2

# 创建背景减除器对象
fgbg = cv2.createBackgroundSubtractorMOG2(history=50, varThreshold=20, detectShadows=False)
# fgbg = cv2.createBackgroundSubtractorKNN()
# mog2 = cv2.createBackgroundSubtractorMOG2(history=50, varThreshold=20, detectShadows=False)
# knn = cv2.createBackgroundSubtractorKNN(history=200, dist2Threshold=200.0, detectShadows=False)


# 打开视频文件
# cap = cv2.VideoCapture('sample_9.mp4')
cap = cv2.VideoCapture('./data/v1.6.2---fast.avi')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 应用背景减除器
    fgmask = fgbg.apply(frame)
    print(fgmask)
    

    # 显示原始帧和前景掩码
    cv2.imshow('Original Frame', frame)
    cv2.imshow('Foreground Mask', fgmask)
    

    # 按 'q' 键退出
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()
    