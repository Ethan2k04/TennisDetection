import sys
import time
import json
import cv2
import os
import threading
import uuid
import urllib.parse
import multiprocessing as mp
import numpy as np

from network import push_data, network_check,push_data_worker,push_data_async
from kernel import (
    detect_target, is_target_result_valid, build_target_status,
    update_target_status, draw_target_boxes, draw_ball_boxes, check_target_status
)
from tools import save_target_to_config, log_with_timestamp
from py_utils.coco_utils import COCO_test_helper
from yolo11 import setup_model, post_process
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


# 获取香橙派设备的MAC地址
mac = uuid.getnode()
mac_address = ':'.join(f'{(mac >> ele) & 0xff:02x}' for ele in range(40, -1, -8))

manager = mp.Manager()
is_ball_in_target = manager.Value('b', False)


def adjust_brightness(frame):
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge((l, a, b))
        frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        return frame


# 目标管理类
class TargetManager:
    def __init__(self):
        self.num_target = 0
        self.target_data = {}
        self.is_target_set = True
        self.force_retarget = False
        self.last_relocate_time = time.time()
        self.target_saved_time = time.strftime('%Y-%m-%d %H:%M:%S')
        if os.path.exists(SETTINGS_FILE):
            with open(SETTINGS_FILE, 'r') as file:
                settings = json.load(file)
                self.num_target = len(settings["target_score"])
        else:
            self.num_target = 0

    def relocate_target(self, frame, retarget_wait_sec: float) -> bool:
        if (not self.is_target_set or self.force_retarget) and \
                time.time() - self.last_relocate_time > retarget_wait_sec:
	    # 亮度修正
            frame = adjust_brightness(frame)
            target_result = detect_target(frame)
            self.last_relocate_time = time.time()
            if is_target_result_valid(target_result, self.num_target):
                self.target_data = self._parse_target_result(target_result)
                save_target_to_config(self.target_data)
                self.target_saved_time = time.strftime('%Y-%m-%d %H:%M:%S')
                log_with_timestamp(
                    f"\033[92m[Valid] Target saved at {self.target_saved_time}\033[0m"
                )
                self.is_target_set = True
                self.force_retarget = False

                return True
            else:
                self.target_data = self._parse_target_result(target_result)
                save_target_to_config(self.target_data)
                log_with_timestamp(
                    "\033[93m[Invalid] No valid target detected. Retrying...\033[0m"
                )
                self.is_target_set = False

                return False

    def _parse_target_result(self, target_result: dict) -> dict:
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
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        self.cap.set(cv2.CAP_PROP_FPS, 60)
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
        self.task_id = 0
        self.ack = 0
        self.reorder_buffer = {}
        self.frame_counter = 0
        self.last_fps = 0
        self.last_fps_calc_time = time.time()

        if not self.cap.isOpened():
            raise RuntimeError(f"Unable to access input source: {input_source}")
        
        with open(CONFIG_FILE, 'r') as file:
            self.config = json.load(file)
            self.target_status = build_target_status(self.config)

        if self.output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(
                self.output_path, fourcc, self.fps, (self.frame_width, self.frame_height)
            )
        else:
            self.video_writer = None

    def process_stream(self, frame_queue, result_queue) -> None:
        while True:
            ret, frame = self.cap.read()
            self.ball_hit_sec, self.retarget_sec = (BALL_HIT_WAIT_SEC, RETARGET_WAIT_SEC)
            if frame is not None:
                raw_frame = frame.copy()

            if not ret:
                log_with_timestamp("\033[93mEnd of video or failed to grab frame\033[0m")
                break

            if not frame_queue.full():
                frame_queue.put((self.task_id, raw_frame))
                self.task_id = (self.task_id + 1) % MAX_QUEUE
                
            if self.target_manager.relocate_target(frame, retarget_wait_sec=self.retarget_sec):
                with open(CONFIG_FILE, 'r') as file:
                    self.config = json.load(file)
                    self.target_status = build_target_status(self.config)

            frame = self._update_score(frame, result_queue)
            frame = self._display_frame(frame)
            if self.video_writer:
                self.video_writer.write(frame)

            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break
            elif key & 0xFF == ord('h'):
                self.target_manager.force_retarget = True

        self._cleanup()

    def _update_score(self, frame, result_queue) -> cv2.Mat:
        if not result_queue.empty():
            task_id, ball_positions = result_queue.get()
            self.reorder_buffer[task_id] = ball_positions
            while self.ack in self.reorder_buffer:
                ball_positions = self.reorder_buffer.pop(self.ack)
                frame = self._process_ball_positions(frame, ball_positions)
                self.ack = (self.ack + 1) % MAX_QUEUE

        frame = draw_target_boxes(frame, self.config)
        return frame

    def _process_ball_positions(self, frame, ball_positions) -> cv2.Mat:
        frame = draw_ball_boxes(frame, ball_positions)
        if len(self.target_status.keys()) > 0:
            ball_center = (0, 0)
            _is_ball_in_target = False
            for _, ball in enumerate(ball_positions):
                ball_center = (int((ball[0] + ball[2]) / 2), int((ball[1] + ball[3]) / 2))
                _is_ball_in_target = update_target_status(self.target_status, ball_center)
                if _is_ball_in_target:
                    is_ball_in_target.set(True)

            if not _is_ball_in_target:
                is_ball_in_target.set(False)

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
                    # push_thread = threading.Thread(
                    #     target=push_data,
                    #     args=(score_data, MAX_RETRY, RETRY_INTERVAL)
                    # )
                    # push_thread.start()
                    push_data_async(score_data)


        return frame

    def _display_frame(self, frame) -> cv2.Mat:
        # 获取当前屏幕的宽度和高度
        frame_width = frame.shape[1]
        frame_height = frame.shape[0]
        frame_scale = frame_width / DEFAULT_FRAME_WIDTH
        if frame is not None:
            frame_display = frame.copy()

        # 根据比例系数计算文字的位置
        title_org = (int(frame_width * TITLE_ORG_RATIO[0]), int(frame_height * TITLE_ORG_RATIO[1]))
        score_org = (int(frame_width * SCORE_ORG_RATIO[0]), int(frame_height * SCORE_ORG_RATIO[1]))
        fps_org = (int(frame_width * FPS_ORG_RATIO[0]), int(frame_height * FPS_ORG_RATIO[1]))
        hint_1_org = (int(frame_width * HINT_1_ORG_RATIO[0]), int(frame_height * HINT_1_ORG_RATIO[1]))
        hint_2_org = (int(frame_width * HINT_2_ORG_RATIO[0]), int(frame_height * HINT_2_ORG_RATIO[1]))
        log_valid_org = (int(frame_width * LOG_VALID_ORG_RATIO[0]), int(frame_height * LOG_VALID_ORG_RATIO[1]))
        log_invalid_org = (int(frame_width * LOG_INVALID_ORG_RATIO[0]), int(frame_height * LOG_INVALID_ORG_RATIO[1]))
        
        # 统计1秒内的帧数
        current_time = time.time()
        self.frame_counter += 1
        
        # 如果距离上一次计算FPS的时间超过1秒，则计算FPS并重置计数器
        frame_rate = self.frame_counter
        if current_time - self.last_fps_calc_time >= 1.0:
            self.last_fps = self.frame_counter
            self.frame_counter = 0
            self.last_fps_calc_time = current_time
        else:
            frame_rate = self.last_fps
        
        # 绘制各种信息
        cv2.putText(
            frame_display, "TENNISv1.1", title_org, cv2.FONT_HERSHEY_SIMPLEX,
            TITLE_SCALE * frame_scale, TITLE_COLOR, int(TITLE_THICKNESS * frame_scale)
        )
        cv2.putText(
            frame_display, f"Score: {self.score_player}", score_org, cv2.FONT_HERSHEY_SIMPLEX,
            SCORE_SCALE * frame_scale, SCORE_COLOR, int(SCORE_THICKNESS * frame_scale)
        )
        cv2.putText(
            frame_display, f"FPS: {frame_rate}", fps_org, cv2.FONT_HERSHEY_SIMPLEX,
            FPS_SCALE * frame_scale, FPS_COLOR, int(FPS_THICKNESS * frame_scale)
        )
        cv2.putText(
            frame_display, f"Press H to retarget", hint_1_org, cv2.FONT_HERSHEY_SIMPLEX,
            HINT_SCALE * frame_scale, HINT_COLOR, int(HINT_THICKNESS * frame_scale)
        )
        cv2.putText(
            frame_display, f"Press Q to quit", hint_2_org, cv2.FONT_HERSHEY_SIMPLEX,
            HINT_SCALE * frame_scale, HINT_COLOR, int(HINT_THICKNESS * frame_scale)
        )
        if self.target_manager.is_target_set:
            cv2.putText(
                frame_display, f"[Valid] Target saved at {self.target_manager.target_saved_time}",
                log_valid_org, cv2.FONT_HERSHEY_SIMPLEX, LOG_SCALE * frame_scale,
                LOG_VALID_COLOR, int(LOG_THICKNESS * frame_scale)
            )
        else:
            cv2.putText(
                frame_display, f"[Invalid] No valid target detected. Retrying...",
                log_invalid_org, cv2.FONT_HERSHEY_SIMPLEX, LOG_SCALE * frame_scale,
                LOG_INVALID_COLOR, int(LOG_THICKNESS * frame_scale)
            )
        cv2.imshow("Video Detection", frame_display)

        return frame

    def _cleanup(self) -> None:
        self.cap.release()
        if self.video_writer:
            self.video_writer.release()

        cv2.destroyAllWindows()


# 摄像头进程
def cam_proc(frame_queue, result_queue):
    # 如果没有传递参数，则默认 input_path 为 "0"
    if len(sys.argv) < 2:
        input_path = "0"  # 默认使用摄像头
        output_path = None
    else:
        input_path = sys.argv[1]
        output_path = None
        if len(sys.argv) > 2:
            output_path = sys.argv[2]

    server_thread = threading.Thread(target=network_check)
    server_thread.daemon = True
    server_thread.start()

    # thread is to push data to server
    push_data_thread = threading.Thread(target=push_data_worker)
    push_data_thread.daemon = True
    push_data_thread.start()

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


# yolo11检测进程
def detect_proc(frame_queue, result_queue):
    def init_model():
        return setup_model(TENNIS_MODEL_PATH)

    def init_co_helper():
        return COCO_test_helper(enable_letter_box=True)

    def process_frame(frame, model, co_helper):
        boxes = []
        ball_positions = []
        ball_conf = BALL_CONF
        img = co_helper.letter_box(im=frame.copy(), new_shape=(IMG_SIZE[1], IMG_SIZE[0]), pad_color=(0, 0, 0))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.expand_dims(img, axis=0)
        outputs = model.run([img])
        if outputs is not None:
            boxes, _, scores = post_process(outputs)

        if boxes is not None:
            boxes = co_helper.get_real_box(boxes)
            for i, box in enumerate(boxes):
                if scores[i] > ball_conf:
                    top, left, right, bottom = box
                    ball_positions.append((int(top), int(left), int(right), int(bottom)))

        return ball_positions
    
    def worker(frame_queue, result_queue):
        model = init_model()
        co_helper = init_co_helper()
        while True:
            task_id, frame = frame_queue.get()
            if frame is None:
                continue
            
            # 亮度修正
            frame = adjust_brightness(frame)

            # 进行检测并推送得分数据
            ball_positions = []
            if is_ball_in_target.get() == False:
                if task_id % FRAME_PER_YOLO == 0:
                    ball_positions = process_frame(frame, model, co_helper)
            else:
                ball_positions = process_frame(frame, model, co_helper)

            if not result_queue.full():
                result_queue.put((task_id, ball_positions))

    # 启动多个进程进行检测
    num_processes = NUM_PROCESSES
    processes = []
    for _ in range(num_processes):
        process = mp.Process(target=worker, args=(frame_queue, result_queue))
        process.daemon = True
        process.start()
        processes.append(process)

    # 等待进程结束
    for process in processes:
        process.join()
        

if __name__ == "__main__":
    frame_queue = mp.Queue(maxsize=MAX_QUEUE)
    result_queue = mp.Queue(maxsize=MAX_QUEUE)
    cam_process = mp.Process(target=cam_proc, args=(frame_queue, result_queue))
    detect_process = mp.Process(target=detect_proc, args=(frame_queue, result_queue))
    cam_process.start()
    detect_process.start()
    cam_process.join()
    detect_process.join()
