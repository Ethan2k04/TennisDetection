import sys
import time

from kernel import *
from constants import *
from tools import save_target_to_config, create_trackbar, get_trackbar_values_wait_sec


# 处理图片的函数
def process_image(image_path):
    # 读取图片
    frame = cv2.imread(image_path)
    if frame is None:
        print("Error: Unable to load image")
        return

    # TODO 处理单张图片

    # 显示处理后的图片
    cv2.imshow("Processed Image", frame)
    cv2.waitKey(0)  # 等待用户按键
    cv2.destroyAllWindows()


# 处理视频的函数
def process_video(video_source):
    # TODO 处理视频
    pass


# 实时视频流处理函数
def process_stream():
    cap = cv2.VideoCapture("test_data/video/t6bn.mp4")  # 0 代表默认摄像头

    if not cap.isOpened():
        print("Error: Unable to access camera")
        return

    score_player = 0
    last_saved_time = time.time()
    ball_timestamps = {}
    target_set = False

    # 创建滑块
    create_trackbar()

    # Get video properties to save the output with the same properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Set up VideoWriter to save the result
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 files
    out = cv2.VideoWriter('result.mp4', fourcc, fps, (frame_width, frame_height))

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame")
            break

        # 从元信息中读取参数
        score_list = []
        num_target = 0
        ball_hit_wait_sec, retarget_wait_sec = get_trackbar_values_wait_sec()
        if os.path.exists(SETTINGS_FILE):
            with open(SETTINGS_FILE, 'r') as file:
                settings = json.load(file)

            score_list = settings["score_list"]
            num_target = settings["num_target"]

        # 定义检测结果
        target_result = {"undef": []}
        for score in score_list:
            target_result[str(score)] = []

        # 检测网球
        ball_result = detect_balls(frame)

        if not target_set:
            # 如果尚未设定好标靶的位置参数，则进行下面的操作
            target_result = find_target_contours(frame)

            target_id = 0
            target_data = {}
            if is_target_result_valid(target_result, num_target):
                # 如果目标符合要求，保存到配置文件
                for score, contours in target_result.items():
                    for contour in contours:
                        (x, y), (major_axis, minor_axis), angle = cv2.fitEllipse(contour)
                        target_data[f"{target_id}"] = {"cls": score, "center_x": x, "center_y": y,
                                                       "major_axis": major_axis, "minor_axis": minor_axis,
                                                       "angle": angle}
                        target_id += 1
                save_target_to_config(target_data)
                print(f"\033[32m[Valid] Target saved to config at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
                target_set = True
                last_saved_time = time.time()
            elif len(target_result["undef"]) > 0:
                # 如果目标不符合要求，全部标记为 undef 并保存到配置文件
                for score, contours in target_result.items():
                    for contour in contours:
                        (x, y), (major_axis, minor_axis), angle = cv2.fitEllipse(contour)
                        target_data[f"{target_id}"] = {"cls": "undef", "center_x": x, "center_y": y,
                                                       "major_axis": major_axis, "minor_axis": minor_axis,
                                                       "angle": angle}
                        target_id += 1
                save_target_to_config(target_data)
                print(f"\033[31m[Undef] Target saved to config at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
                last_saved_time = time.time()

        if target_set and time.time() - last_saved_time > retarget_wait_sec:
            # 如果已设定好标靶位置且 WAIT_SEC 秒已过，设置 target_set = False 并在下一帧更新标靶位置
            target_set = False
            print(f"{retarget_wait_sec}s has passed, detect target again.")

        if os.path.exists(CONFIG_FILE):
            # 读取配置文件
            with open(CONFIG_FILE, 'r') as file:
                config = json.load(file)

            # 根据配置中的目标框绘制
            draw_target_boxes(frame, config)

            # 检测网球并更新分数
            ball_id = 0
            for ball in ball_result:
                ball_center = (int((ball[0] + ball[2]) / 2), int((ball[1] + ball[3]) / 2))
                is_collided, score = detect_collision(ball_center, config)
                if is_collided:
                    if ball_id not in ball_timestamps:
                        ball_timestamps[ball_id] = time.time()  # 记录网球进入标靶区域的时间
                        # TODO 尚未支持多目标的情况
                        # ball_id += 1
                    elif time.time() - ball_timestamps[ball_id] > ball_hit_wait_sec:  # 判断是否已超过2秒
                        score_player += score
                        ball_timestamps.pop(ball_id)  # 防止多次增加分数

        # 显示分数
        cv2.putText(frame, f"Score: {score_player}", SCORE_ORG, cv2.FONT_HERSHEY_SIMPLEX, SCORE_SCALE, SCORE_COLOR,
                    SCORE_THICKNESS)

        # 绘制网球的目标框
        frame = draw_ball_boxes(frame, ball_result)

        # 保存每一帧到视频文件
        out.write(frame)

        # 显示图像
        cv2.imshow("Real-Time Target Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py <video_path or image_path>")
        sys.exit(1)

    input_path = sys.argv[1]

    # 如果输入的是视频文件
    if input_path.endswith(('.mp4', '.avi', '.mov')):
        process_video(input_path)
    # 如果输入的是图片文件
    elif input_path.endswith(('.jpg', '.png', '.jpeg')):
        process_image(input_path)
    # 如果输入为 0，代表实时视频流处理
    elif input_path == "0":
        process_stream()
    else:
        print(f"Error: Unsupported file type or invalid input {input_path}")
        sys.exit(1)


if __name__ == "__main__":
    main()
