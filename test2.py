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


target_manager = TargetManager()
ball_manager = TennisBallManager()

# Open the video file

# cap = cv2.VideoCapture('real_sample_1.mp4')
cap = cv2.VideoCapture('real_sample_2.mp4')

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
    # if count < 1590:
    # fly over white target 1 
    # if count < 3000:
    # real_sample_2
    # if count < 800:
    # if count < 420:
        # continue

   
    if not target_manager.relocate_target(frame, retarget_wait_sec=RETARGET_WAIT_SEC):
        # time.sleep(1)
        continue


    target_roi = target_manager.target_roi.get_roi(frame=frame)

    frame = target_manager.draw_target_region(frame=frame)
    frame = target_manager.draw_target_circles(frame=frame)

    binary_roi = ball_manager.get_green_from_roi(target_roi)

    # in target roi find ball 
    if ball_manager.find_ball(target_roi,count):
        x,y ,w, h = ball_manager.get_last_ball()
        target_manager.draw_ball_in_roi(frame=frame,center_x=x, center_y =y,width= w,height= h)
    
    # ball hit canvas test 
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
    # cv2.imshow('target roi', binary_roi)


    # if count == 100:
    #     cv2.waitKey(0)
    #     continue


    # Wait indefinitely for a key press to continue to the next frame
    key = cv2.waitKey(1) & 0xFF
    # print(key)
    if key == ord('q'):
        break
    elif key == ord('h'):
        target_manager.force_retarget = True


# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
