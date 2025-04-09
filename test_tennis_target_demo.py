import cv2
import matplotlib.pyplot as plt
import numpy as np
from test_tennis_ball_util import make_binary_bitmap, find_tenis_ball,TenisBall, make_binary_bitmap_from_frame, tennis_ball_hit_test
from typing import List

# Open the video file
# cap = cv2.VideoCapture('test3.mp4')
cap = cv2.VideoCapture('sample_1.mp4')
# cap = cv2.VideoCapture('sample_2.mp4')

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

# Initialize the plots
# plt.ion()  # Turn on interactive mode
# fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

# # Initialize with dummy images
# cax1 = ax1.imshow(np.zeros((480, 640)), cmap='hsv', norm=None)
# cax2 = ax2.imshow(np.zeros((480, 640)), cmap='gray', norm=None)
# cax3 = ax3.imshow(np.zeros((480, 640)), cmap='gray', norm=None)

# # Add colorbars
# fig.colorbar(cax1, ax=ax1)
# fig.colorbar(cax2, ax=ax2)
# fig.colorbar(cax3, ax=ax3)

# # Set titles
# ax1.set_title('Hue Channel Heatmap')
# ax2.set_title('Value Channel Heatmap')
# ax3.set_title('Binary Bitmap')


count = 0
# loop read frame
last_tennis_ball_count = 0 
hit_count  = 0
tennis_ball_list : List[TenisBall] = []


while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    count += 1
    print('count = ', count)

    # if count % 3 != 0 :
    #     continue

    # save file to find scope 
    # if count == 100 :
    #     cv2.imwrite('test.jpg', frame)

    # skip for sample_1
    # 171, 573
    # if count < 570 :
    #     continue

    # if count < 160 :
    #     continue

    if count < 950 :
        continue

    # skip for sample_2
    # if count < 230 :
    #     continue

    # if count < 1390 :
    #     continue

    # if count < 560 :
    #     continue

    # Convert the frame to HSV format
    # hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Split the HSV frame into its channels
    # h, s, v = cv2.split(hsv_frame)

    # Apply a threshold to the value channel to create a binary bitmap
    # _, binary_bitmap = cv2.threshold(v, 128, 1, cv2.THRESH_BINARY)
    # binary_bitmap = make_binary_bitmap(h,s,v)

    binary_bitmap = make_binary_bitmap_from_frame(frame)

    found, tennis_ball  = find_tenis_ball(binary_bitmap,count)
    if found:

        # print(repr(tennis_ball))

        # Draw a bounding box around the detected tennis ball
        top_left = (tennis_ball.centerx - tennis_ball.width // 2, tennis_ball.centery - tennis_ball.height // 2)
        bottom_right = (tennis_ball.centerx + tennis_ball.width // 2, tennis_ball.centery + tennis_ball.height // 2)
      # 2 seconds, new ball list
        if count - last_tennis_ball_count > 120:
            hit_count = 0
            tennis_ball_list.clear()
        
        last_tennis_ball_count = count

        if len(tennis_ball_list) > 0 :
            last = tennis_ball_list[-1]
            tennis_ball.calculate_v_a(last)

        tennis_ball_list.append(tennis_ball)
        print(tennis_ball.show_v_cross())
        # print(tennis_ball.show_v_a())
       
        if tennis_ball_hit_test(tennis_ball_list):
            hit_count += 1

        if hit_count == 1:
            # first time hit show red
            cv2.rectangle(frame, top_left, bottom_right, (0, 0, 255), 2)  # hit red
            # trick only show once
            hit_count += 1 
            print("*********hit : {count}")
        else:
            cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)  # not hit green 
      

    # Update the heatmaps without normalization
    # cax1.set_data(h)
    # cax1.set_clim(0, 179)  # Set the color limits to the range of hue values


    # cax2.set_data(v)
    # cax2.set_clim(0, 255)  # Set the color limits to the range of value values

    # cax3.set_data(binary_bitmap)
    # cax3.set_clim(0, 1)  # Set the color limits to the range of binary values

    # plt.draw()
    # plt.pause(0.001)  # Pause to allow the plot to update

    # # Display the hue channel and binary bitmap using OpenCV (optional)
    # cv2.imshow('Hue Channel', h)
    # cv2.imshow('Binary Bitmap', binary_bitmap * 255)  # Scale to 0-255 for display

    cv2.imshow('Tennis Ball', frame)
    # cv2.imshow('Tennis Ball', binary_bitmap * 255 )
    
    # Wait indefinitely for a key press to continue to the next frame
    key = cv2.waitKey(1)
    if key == ord('q'):
        break


# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
# plt.ioff()  # Turn off interactive mode
# plt.show()  # Keep the last plot open