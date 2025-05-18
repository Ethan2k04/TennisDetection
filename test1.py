import cv2
import matplotlib.pyplot as plt
import numpy as np
from typing import List

# 初始化 matplotlib 图形
plt.ion()  # 开启交互模式
fig, ax = plt.subplots()
line, = ax.plot(np.arange(180), np.zeros((180,)), 'b-')
ax.set_xlim(0, 180)
ax.set_ylim(0, 10000)
ax.set_xlabel('Hue Value')
ax.set_ylabel('Frequency')
ax.set_title('Hue Histogram')

# Open the video file

frame_rate = 1 / 60 
print(frame_rate)
cap = cv2.VideoCapture('real_sample_1.mp4')
# cap = cv2.VideoCapture('real_sample_2.mp4')

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()


# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_rate = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = frame_count / frame_rate

print(f"Frame Width: {frame_width}")
print(f"Frame Height: {frame_height}")
print(f"Frame Rate: {frame_rate} FPS")
print(f"Total Frames: {frame_count}")
print(f"Duration: {duration} seconds")


count = 0


while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    count += 1
    print('count = ', count)

  
    # real_sample_1
    if count < 420:
        continue

     # 将帧转换为 HSV 颜色空间
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # 提取色调通道
    hue_channel = hsv_frame[:, :, 0]
    hist = cv2.calcHist([hue_channel], [0], None, [180], [0, 180])
    
    
  # 更新 matplotlib 图形
    line.set_ydata(hist)
    fig.canvas.draw()
    plt.pause(0.01)


    # 显示结果
    cv2.imshow('Video with Histogram', frame)
  
    
    # Wait indefinitely for a key press to continue to the next frame
    key = cv2.waitKey(0) & 0xFF
    # print(key)
    if key == ord('q'):
        break


# 关闭 matplotlib 图形
plt.ioff()
plt.close()

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
# plt.ioff()  # Turn off interactive mode
# plt.show()  # Keep the last plot open
