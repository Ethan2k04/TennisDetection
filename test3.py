import cv2
import numpy as np

# 打开视频文件或摄像头
cap = cv2.VideoCapture('real_sample_4.mp4')

# 读取前三帧
ret, frame1 = cap.read()
ret, frame2 = cap.read()
ret, frame3 = cap.read()

# 转换为灰度图并进行高斯滤波
gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
gray1 = cv2.GaussianBlur(gray1, (21, 21), 0)
gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
gray2 = cv2.GaussianBlur(gray2, (21, 21), 0)
gray3 = cv2.cvtColor(frame3, cv2.COLOR_BGR2GRAY)
gray3 = cv2.GaussianBlur(gray3, (21, 21), 0)

count = 3
while True:
    # 计算第一帧和第二帧的差值
    diff1 = cv2.absdiff(gray1, gray2)
    # 计算第二帧和第三帧的差值
    diff2 = cv2.absdiff(gray2, gray3)

    # 二值化处理
    thresh1 = cv2.threshold(diff1, 25, 255, cv2.THRESH_BINARY)[1]
    thresh2 = cv2.threshold(diff2, 25, 255, cv2.THRESH_BINARY)[1]

    # 逻辑与运算
    thresh = cv2.bitwise_and(thresh1, thresh2)

    # 形态学操作：膨胀和腐蚀
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=2)
    thresh = cv2.erode(thresh, kernel, iterations=1)



    # 查找轮廓
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) < 500:
            continue
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame3, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # 显示结果
    cv2.imshow("Frame", frame3)
    cv2.imshow("Thresh", thresh)

    # 更新帧
    frame1 = frame2
    frame2 = frame3
    ret, frame3 = cap.read()
    if not ret:
        break

    # 更新灰度图
    gray1 = gray2
    gray2 = gray3
    gray3 = cv2.cvtColor(frame3, cv2.COLOR_BGR2GRAY)
    gray3 = cv2.GaussianBlur(gray3, (21, 21), 0)

    count = count + 3
    print(count)

    # 按 'q' 键退出循环
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break

# 释放资源并关闭窗口
cap.release()
cv2.destroyAllWindows()
    