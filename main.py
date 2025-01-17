import sys
import time
import json
import cv2
import os
import threading
import uuid
import urllib.parse
from network import push_data, network_check
from kernel import detect_target, is_target_result_valid, build_target_status, update_target_status, \
    draw_target_boxes, draw_ball_boxes, check_target_status
from tools import create_trackbar, save_target_to_config, log_with_timestamp, get_trackbar_values_wait_sec
from constants import BALL_HIT_WAIT_SEC, CONFIG_FILE, FPS_COLOR, FPS_SCALE, FPS_THICKNESS, \
    HINT_COLOR, HINT_SCALE, HINT_THICKNESS, LOG_INVALID_COLOR, \
    LOG_SCALE, LOG_THICKNESS, LOG_VALID_COLOR, MAX_RETRY, RETARGET_WAIT_SEC, RETRY_INTERVAL, \
    SCORE_COLOR, SCORE_SCALE, SCORE_THICKNESS, SETTINGS_FILE, TITLE_COLOR, TITLE_SCALE, \
    TITLE_THICKNESS, X_COOR_WEIGHT, Y_COOR_WEIGHT, TITLE_ORG_RATIO, SCORE_ORG_RATIO, FPS_ORG_RATIO, HINT_1_ORG_RATIO, \
    HINT_2_ORG_RATIO, HINT_3_ORG_RATIO, LOG_INVALID_ORG_RATIO, LOG_VALID_ORG_RATIO, DEFAULT_FRAME_WIDTH, IMG_SIZE, TENNIS_MODEL_PATH
import multiprocessing as mp
import numpy as np
from py_utils.coco_utils import COCO_test_helper
from yolo11 import setup_model, post_process

# 获取香橙派设备的MAC地址
mac = uuid.getnode()
mac_address = ':'.join(f'{(mac >> ele) & 0xff:02x}' for ele in range(40, -1, -8))


# 目标管理类
class TargetManager:
    def __init__(self):
        self.is_target_set = True
        self.last_relocate_time = time.time()
        self.target_saved_time = time.strftime('%Y-%m-%d %H:%M:%S')
        self.num_target = 0
        self.target_data = {}
        self.force_retarget = False
        self.debug = False
        if os.path.exists(SETTINGS_FILE):
            with open(SETTINGS_FILE, 'r') as file:
                settings = json.load(file)
                self.num_target = settings["num_target"]

    def relocate_target(self, frame, retarget_wait_sec: float) -> bool:
        if (not self.is_target_set or self.force_retarget) and time.time() - self.last_relocate_time > retarget_wait_sec:
            target_result = detect_target(frame, self.debug)
            self.last_relocate_time = time.time()
            if is_target_result_valid(target_result, self.num_target):
                self.target_data = self._parse_target_result(target_result)
                save_target_to_config(self.target_data)
                self.target_saved_time = time.strftime('%Y-%m-%d %H:%M:%S')
                log_with_timestamp(f"\033[92m[Valid] Target saved at {self.target_saved_time}\033[0m")
                self.is_target_set = True
                self.force_retarget = False
                self.debug = False
                return True
            else:
                self.target_data = self._parse_target_result(target_result)
                save_target_to_config(self.target_data)
                log_with_timestamp(f"\033[93m[Invalid] No valid target detected. Retrying...\033[0m")
                self.is_target_set = False
                return False

    @staticmethod
    def _parse_target_result(target_result: dict) -> dict:
        target_data = []
        target_id = 0

        # 将靶标检测结果转化为临时json格式
        for score, contours in target_result.items():
            for contour in contours:
                (x, y), (major_axis, minor_axis), angle = cv2.fitEllipse(contour)
                score_value = X_COOR_WEIGHT * x + Y_COOR_WEIGHT * y
                target_data.append({
                    "id": target_id,
                    "cls": score,
                    "center_x": x,
                    "center_y": y,
                    "major_axis": major_axis,
                    "minor_axis": minor_axis,
                    "angle": angle,
                    "score_value": score_value,
                })
                target_id += 1

        # 对靶标根据score_value进行排序（以实现从上到下从左到右编号）
        target_data.sort(key=lambda item: item["score_value"])

        # 转化为结果json格式
        sorted_target_data = {
            str(idx + 1): {
                "cls": target["cls"],
                "center_x": target["center_x"],
                "center_y": target["center_y"],
                "major_axis": target["major_axis"],
                "minor_axis": target["minor_axis"],
                "angle": target["angle"],
            }
            for idx, target in enumerate(target_data)
        }

        return sorted_target_data


# 视频处理类
class VideoProcessor:
    def __init__(self, input_source=0, output_path=None):
        self.cap = cv2.VideoCapture(input_source)

        # ---{解决帧率问题开始}---
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        self.cap.set(cv2.CAP_PROP_FPS, 60)
        # ---{解决帧率问题结束}---

        if not self.cap.isOpened():
            raise RuntimeError(f"Unable to access input source: {input_source}")

        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        self.output_path = output_path
        self.target_manager = TargetManager()
        self.score_player = 0
        self.last_frame_time = time.time()
        self.ball_timestamps = {}
        self.target_status = {}
        self.config = {}
        self.last_collision_time = time.time()
        self.ball_hit_sec = BALL_HIT_WAIT_SEC
        self.retarget_sec = RETARGET_WAIT_SEC
        with open(CONFIG_FILE, 'r') as file:
            self.config = json.load(file)
            self.target_status = build_target_status(self.config)
        if self.output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(self.output_path, fourcc, self.fps, (self.frame_width, self.frame_height))
        else:
            self.video_writer = None

    def process_stream(self, frame_queue, result_queue) -> None:
        create_trackbar()
        while True:
            ret, frame = self.cap.read()
            self.ball_hit_sec, self.retarget_sec = get_trackbar_values_wait_sec()
            if not ret:
                log_with_timestamp("\033[93mEnd of video or failed to grab frame\033[0m")
                break

            frame_queue.put(frame)

            if self.target_manager.relocate_target(frame, retarget_wait_sec=self.retarget_sec):
                with open(CONFIG_FILE, 'r') as file:
                    self.config = json.load(file)
                    self.target_status = build_target_status(self.config)

            frame = self._update_score(frame, frame_queue, result_queue)
            frame = self._display_frame(frame)

            if self.video_writer:
                self.video_writer.write(frame)

            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break
            elif key & 0xFF == ord('h'):
                self.target_manager.force_retarget = True
            elif key & 0xFF == ord('j'):
                self.target_manager.debug = True

        self._cleanup()

    def _update_score(self, frame, frame_queue, result_queue) -> cv2.Mat:
        ball_result = []
        ball_result = result_queue.get()

        frame = draw_ball_boxes(frame, ball_result)
        frame = draw_target_boxes(frame, self.config)

        if len(self.target_status.keys()) > 0:
            ball_center = (0, 0)
            for _, ball in enumerate(ball_result):
                ball_center = (int((ball[0] + ball[2]) / 2), int((ball[1] + ball[3]) / 2))
                update_target_status(self.target_status, ball_center)

            collision_detected, score, idx = check_target_status(self.target_status, frame)
            if collision_detected and abs(time.time() - self.last_collision_time > self.ball_hit_sec):
                if score != 0:
                    self.last_collision_time = time.time()
                    self.score_player += score
                    encoded_mac = urllib.parse.quote(mac_address)
                    score_data = {
                        "x": ball_center[0],
                        "y": ball_center[1],
                        "score": score,
                        "device_id": encoded_mac,
                        "target_id": idx,
                    }
                    # 推送得分数据
                    push_thread = threading.Thread(
                        target=push_data,
                        args=(score_data, MAX_RETRY, RETRY_INTERVAL)
                    )
                    push_thread.start()

        return frame

    def _display_frame(self, frame) -> cv2.Mat:
        # 获取当前屏幕的宽度和高度
        frame_width = frame.shape[1]
        frame_height = frame.shape[0]
        frame_scale = frame_width / DEFAULT_FRAME_WIDTH

        frame_cpy = frame.copy()

        # 根据比例系数计算文字的位置
        title_org = (int(frame_width * TITLE_ORG_RATIO[0]), int(frame_height * TITLE_ORG_RATIO[1]))
        score_org = (int(frame_width * SCORE_ORG_RATIO[0]), int(frame_height * SCORE_ORG_RATIO[1]))
        fps_org = (int(frame_width * FPS_ORG_RATIO[0]), int(frame_height * FPS_ORG_RATIO[1]))
        hint_1_org = (int(frame_width * HINT_1_ORG_RATIO[0]), int(frame_height * HINT_1_ORG_RATIO[1]))
        hint_2_org = (int(frame_width * HINT_2_ORG_RATIO[0]), int(frame_height * HINT_2_ORG_RATIO[1]))
        hint_3_org = (int(frame_width * HINT_3_ORG_RATIO[0]), int(frame_height * HINT_3_ORG_RATIO[1]))
        log_valid_org = (int(frame_width * LOG_VALID_ORG_RATIO[0]), int(frame_height * LOG_VALID_ORG_RATIO[1]))
        log_invalid_org = (int(frame_width * LOG_INVALID_ORG_RATIO[0]), int(frame_height * LOG_INVALID_ORG_RATIO[1]))

        # 获取当前时间和帧率
        current_time = time.time()
        frame_rate = round(1 / (current_time - self.last_frame_time))
        self.last_frame_time = current_time

        # 绘制各种信息
        cv2.putText(frame_cpy, "TENNISv1.0", title_org, cv2.FONT_HERSHEY_SIMPLEX, TITLE_SCALE * frame_scale, TITLE_COLOR, int(TITLE_THICKNESS * frame_scale))
        cv2.putText(frame_cpy, f"Score: {self.score_player}", score_org, cv2.FONT_HERSHEY_SIMPLEX, SCORE_SCALE * frame_scale, SCORE_COLOR, int(SCORE_THICKNESS * frame_scale))
        cv2.putText(frame_cpy, f"FPS: {frame_rate}", fps_org, cv2.FONT_HERSHEY_SIMPLEX, FPS_SCALE * frame_scale, FPS_COLOR, int(FPS_THICKNESS * frame_scale))
        cv2.putText(frame_cpy, f"Press H to retarget", hint_1_org, cv2.FONT_HERSHEY_SIMPLEX, HINT_SCALE * frame_scale, HINT_COLOR, int(HINT_THICKNESS * frame_scale))
        cv2.putText(frame_cpy, f"Press J to show mask", hint_2_org, cv2.FONT_HERSHEY_SIMPLEX, HINT_SCALE * frame_scale, HINT_COLOR, int(HINT_THICKNESS * frame_scale))
        cv2.putText(frame_cpy, f"Press Q to quit", hint_3_org, cv2.FONT_HERSHEY_SIMPLEX, HINT_SCALE * frame_scale, HINT_COLOR, int(HINT_THICKNESS * frame_scale))
        
        if self.target_manager.is_target_set:
            cv2.putText(frame_cpy, f"[Valid] Target saved at {self.target_manager.target_saved_time}", log_valid_org, cv2.FONT_HERSHEY_SIMPLEX, LOG_SCALE * frame_scale, LOG_VALID_COLOR, int(LOG_THICKNESS * frame_scale))
        else:
            cv2.putText(frame_cpy, f"[Invalid] No valid target detected. Retrying...", log_invalid_org, cv2.FONT_HERSHEY_SIMPLEX, LOG_SCALE * frame_scale, LOG_INVALID_COLOR, int(LOG_THICKNESS * frame_scale))

        cv2.imshow("Video Detection", frame_cpy)

        return frame


    def _cleanup(self) -> None:
        self.cap.release()
        if self.video_writer:
            self.video_writer.release()
        cv2.destroyAllWindows()


# 主函数
def main_proc(frame_queue, result_queue):
    if len(sys.argv) < 2:
        print("Usage: python3 main.py <video_path or image_path or 0 for stream> [output_path (optional)]")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = None

    if len(sys.argv) > 2:
        output_path = sys.argv[2]

    server_thread = threading.Thread(target=network_check)
    server_thread.daemon = True
    server_thread.start()

    if input_path.endswith(('.mp4', '.avi', '.mov')):
        if output_path:
            processor = VideoProcessor(input_source=input_path, output_path=output_path)
        else:
            processor = VideoProcessor(input_source=input_path)
        processor.process_stream(frame_queue, result_queue)
    elif input_path.endswith(('.jpg', '.png', '.jpeg')):
        print("Image processing not implemented yet")
    elif input_path == "0":
        processor = VideoProcessor(input_source=0)
        processor.process_stream(frame_queue, result_queue)
    else:
        print(f"Error: Unsupported file type or invalid input {input_path}")
        sys.exit(1)


def detect_proc(frame_queue, result_queue):
    model_tennis = setup_model(TENNIS_MODEL_PATH)
    co_helper = COCO_test_helper(enable_letter_box=True)
    while True:
        frame = frame_queue.get()
        ball_conf = 0.1 # get_trackbar_values_confidence()
        img = co_helper.letter_box(im=frame.copy(), new_shape=(IMG_SIZE[1], IMG_SIZE[0]), pad_color=(0, 0, 0))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.expand_dims(img, axis=0)
        outputs = model_tennis.run([img])
        boxes = []
        if outputs is not None:
            boxes, _, scores = post_process(outputs)

        ball_positions = []
        if boxes is not None:
            boxes = co_helper.get_real_box(boxes)
            for i, box in enumerate(boxes):
                if scores[i] > ball_conf:
                    top, left, right, bottom = box
                    ball_positions.append((int(top), int(left), int(right), int(bottom)))

        # 把结果存储到结果队列
        result_queue.put(ball_positions)


if __name__ == "__main__":
    frame_queue = mp.Queue()
    result_queue = mp.Queue()
    
    main_process = mp.Process(target=main_proc, args=(frame_queue, result_queue))
    main_process.daemon = True
    detect_process = mp.Process(target=detect_proc, args=(frame_queue, result_queue))

    main_process.start()
    detect_process.start()

    main_process.join()
    detect_process.join()
