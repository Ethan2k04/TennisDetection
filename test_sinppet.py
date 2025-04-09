import cv2
import numpy as np

# 读取图像
image = cv2.imread('your_image.jpg')

# 将图像从 BGR 颜色空间转换为 HSV 颜色空间
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# 定义绿色在 HSV 颜色空间中的范围
# 注意：OpenCV 中 HSV 的 H 范围是 0 - 180，S 和 V 范围是 0 - 255
lower_green = np.array([40, 40, 40])
upper_green = np.array([80, 255, 255])

# 创建掩码，提取绿色区域
mask = cv2.inRange(hsv, lower_green, upper_green)

# 将掩码应用到原始图像上，提取绿色部分
result = cv2.bitwise_and(image, image, mask=mask)

# 显示原始图像、掩码和结果图像
cv2.imshow('Original Image', image)
cv2.imshow('Green Mask', mask)
cv2.imshow('Green Region', result)

# 等待按键事件，按任意键关闭所有窗口
cv2.waitKey(0)
cv2.destroyAllWindows()



