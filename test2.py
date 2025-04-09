import cv2
import matplotlib.pyplot as plt
import numpy as np
from typing import List
from pprint import pprint
import time
from target_manager import TargetManager
from constants import (
    BALL_HIT_WAIT_SEC, CONFIG_FILE, FPS_COLOR, FPS_SCALE, FPS_THICKNESS,
    HINT_COLOR, HINT_SCALE, HINT_THICKNESS, LOG_INVALID_COLOR, LOG_SCALE,
    LOG_THICKNESS, LOG_VALID_COLOR, MAX_RETRY, RETARGET_WAIT_SEC,
    RETRY_INTERVAL, SCORE_COLOR, SCORE_SCALE, SCORE_THICKNESS, SETTINGS_FILE,
    TITLE_COLOR, TITLE_SCALE, TITLE_THICKNESS, X_COOR_WEIGHT, Y_COOR_WEIGHT,
    TITLE_ORG_RATIO, SCORE_ORG_RATIO, FPS_ORG_RATIO, HINT_1_ORG_RATIO,
    HINT_2_ORG_RATIO, LOG_INVALID_ORG_RATIO, LOG_VALID_ORG_RATIO,
    DEFAULT_FRAME_WIDTH, IMG_SIZE, TENNIS_MODEL_PATH, BALL_CONF, NUM_PROCESSES,
    MAX_QUEUE, FRAME_PER_YOLO
)

from tennis_ball_manager import TennisBallManager


def _adjust_brightness(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return frame



# plt.ion()  # 开启交互模式
# fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(16, 8))
# axs[0].set_xlim([0, 179])
# axs[0].set_title(" hue Histogram")

# axs[1].set_xlim([0, 255])
# axs[1].set_title(" saturation Histogram")

# axs[2].set_xlim([0, 255])
# axs[2].set_title("value Histogram")

target_manager = TargetManager()
ball_manager = TennisBallManager()

# Open the video file

# cap = cv2.VideoCapture('real_sample_1.mp4')
# cap = cv2.VideoCapture('real_sample_2.mp4')
# cap = cv2.VideoCapture('real_sample_3.mp4')
cap = cv2.VideoCapture('real_sample_7.mp4')
# cap = cv2.VideoCapture('real_sample_8.mp4')

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
    # if count < 430:
    # if count < 1590:
    # fly over white target 1 
    # if count < 3000:
    # real_sample_2
    #black score 10
    # if count < 1590:
    # if count < 800:
    # if count < 420:
        # continue
        
    # if count < 220:
    #     continue

    # real_sample_7 
    if count < 1994:
        continue

   
    if not target_manager.relocate_target(frame, retarget_wait_sec=RETARGET_WAIT_SEC):
        # time.sleep(1)
        continue


    target_roi = target_manager.target_roi.get_roi(frame=frame)
    # target_roi = _adjust_brightness(target_roi)


    target_roi_black_10 = target_manager.target_roi_black_10.get_roi(frame=frame)
    target_manager.set_target_black_10_bg_hsv(target_roi_black_10)
    binary_image = target_manager.get_ball_like_in_black_score_10(target_roi_black_10)
    ball,ball_info = target_manager.find_ball_in_black_score_10(binary_image, target_roi_black_10,count)
    if ball :
        print(ball.show_v_dot())
        ball_manager.add_ball(ball=ball)

        # ball_manager.set_tennis_ball_info(ball_info=ball_info)
        ball_manager.add_tennis_ball_info(ball_info=ball_info)
        
        x,y ,w, h = ball_manager.get_last_ball()
        # print(f'x={x} y={y} w={w} h={h}')
        target_manager.draw_ball_in_roi(frame=frame,center_x=x, center_y =y,width= w,height= h)
    else:
        frame = target_manager.draw_target_region(frame=frame)
        frame = target_manager.draw_target_circles(frame=frame)

        binary_roi = ball_manager.get_green_from_roi(target_roi)
        cv2.imshow('target roi', binary_roi)
    # binary_roi = ball_manager.get_green_from_roi_background(target_roi)


    # h_roi = target_manager.get_hue_from_roi(target_roi_black_10)
    # h_hist = target_manager.get_hue_hist_from_roi(target_roi_black_10)

    # s_roi = target_manager.get_s_from_roi(target_roi_black_10)
    # s_hist = target_manager.get_s_hist_from_roi(target_roi_black_10)

    # v_roi = target_manager.get_v_from_roi(target_roi_black_10)
    # v_hist = target_manager.get_v_hist_from_roi(target_roi_black_10)

    # axs[0].clear()
    # axs[0].plot(h_hist)

    # axs[1].clear()
    # axs[1].plot(s_hist)

    # axs[2].clear()
    # axs[2].plot(v_hist)

    # plt.tight_layout()
    # plt.draw()
    # plt.pause(0.001)  # 短暂暂停以更新画面

    # in target roi find ball 
        if ball_manager.find_ball(target_roi,count):
            x,y ,w, h = ball_manager.get_last_ball()
            target_manager.draw_ball_in_roi(frame=frame,center_x=x, center_y =y,width= w,height= h)
    
    # ball hit canvas test ,网球撞击幕布测试
    hit_result = ball_manager.hit_test(count)
    if hit_result:
        x,y, is_near_white = hit_result
        if target_manager.hit_score_test((x,y), is_near_white=is_near_white):
            # show score in frame and send to server 
            # print(f"********* scores: {target_manager.score_player} ********")
            pass

    # show score on frame
    frame_width = frame.shape[1]
    frame_height = frame.shape[0]
    frame_scale = frame_width / DEFAULT_FRAME_WIDTH

    score_org = (int(frame_width * SCORE_ORG_RATIO[0]), int(frame_height * SCORE_ORG_RATIO[1]))
    cv2.putText(
        frame, f"Score: {target_manager.score_player}", score_org, cv2.FONT_HERSHEY_SIMPLEX,
         SCORE_SCALE * frame_scale, SCORE_COLOR, int(SCORE_THICKNESS * frame_scale)
    )
        
    # print("target set")
    # pprint(target_manager.target_data)
    # 显示结果
    cv2.imshow('Video with Histogram', frame)
    # cv2.imshow('hue roi', h_roi)
    # cv2.imshow('s roi', s_roi)
    # cv2.imshow('v roi', v_roi)
    # cv2.imshow('binary', binary_image)
    # cv2.imshow('target roi', target_roi)
    # cv2.imshow('target roi', binary_roi)



    # if count == 100:
    #     cv2.waitKey(0)
    #     continue


    # Wait indefinitely for a key press to continue to the next frame
    key = cv2.waitKey(0) & 0xFF
    # print(key)
    if key == ord('q'):
        break
    elif key == ord('h'):
        target_manager.force_retarget = True


# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
