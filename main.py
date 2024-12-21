import sys
import time
import json
import cv2
import os
import threading
import uuid
from network import push_data
from kernel import detect_balls, detect_target, is_target_result_valid, build_target_status, update_target_status, \
    draw_target_boxes, draw_ball_boxes, check_target_status
from tools import create_trackbar, save_target_to_config, log_with_timestamp
from constants import CONFIG_FILE, SCORE_ORG, SCORE_SCALE, SCORE_COLOR, SCORE_THICKNESS, FPS_SCALE, FPS_COLOR, \
    FPS_THICKNESS, FPS_ORG, RETARGET_WAIT_SEC, MAX_RETRY, RETRY_INTERVAL, SETTINGS_FILE, BALL_HIT_WAIT_SEC,\
    TITLE_THICKNESS, TITLE_ORG, TITLE_COLOR, TITLE_SCALE, X_COOR_WEIGHT, Y_COOR_WEIGHT, HINT_COLOR, HINT_1_ORG, \
    HINT_2_ORG, HINT_SCALE, HINT_THICKNESS, HINT_3_ORG, LOG_VALID_ORG, LOG_INVALID_ORG, LOG_VALID_COLOR, \
    LOG_INVALID_COLOR, LOG_THICKNESS, LOG_SCALE


# 获取香橙派设备的MAC地址
mac = uuid.getnode()
mac_address = ':'.join(f'{(mac >> ele) & 0xff:02x}' for ele in range(40, -1, -8))


# 目标管理类
class TargetManager:
    def __init__(self):
        self.is_target_set = False
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
        self.last_collision_time = time.time()
        if self.output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(self.output_path, fourcc, self.fps, (self.frame_width, self.frame_height))
        else:
            self.video_writer = None

    def process_stream(self) -> None:
        create_trackbar()
        while True:
            ret, frame = self.cap.read()
            if not ret:
                log_with_timestamp("\033[93mEnd of video or failed to grab frame\033[0m")
                break

            if self.target_manager.relocate_target(frame, retarget_wait_sec=RETARGET_WAIT_SEC):
                with open(CONFIG_FILE, 'r') as file:
                    config = json.load(file)
                    self.target_status = build_target_status(config)

            frame = self._update_score(frame)
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

    def _update_score(self, frame) -> cv2.Mat:
        ball_result = detect_balls(frame)
        with open(CONFIG_FILE, 'r') as file:
            config = json.load(file)

        frame = draw_ball_boxes(frame, ball_result)
        frame = draw_target_boxes(frame, config)

        if len(self.target_status.keys()) > 0:
            ball_center = (0, 0)
            for ball_id, ball in enumerate(ball_result):
                ball_center = (int((ball[0] + ball[2]) / 2), int((ball[1] + ball[3]) / 2))
                update_target_status(self.target_status, ball_center)

            collision_detected, score, idx = check_target_status(self.target_status, frame)
            if collision_detected and abs(time.time() - self.last_collision_time > BALL_HIT_WAIT_SEC):
                if score != 0:
                    self.last_collision_time = time.time()
                    self.score_player += score
                    score_data = {
                        "x": ball_center[0],
                        "y": ball_center[1],
                        "score": score,
                        # TODO: 服务器那边好像目前不能接收这两个字段
                        # "target_id": idx,
                        # "device_id": mac_address
                    }
                    # 推送得分数据
                    push_thread = threading.Thread(
                        target=push_data,
                        args=(score_data, MAX_RETRY, RETRY_INTERVAL)
                    )
                    push_thread.start()

        return frame

    def _display_frame(self, frame) -> cv2.Mat:
        current_time = time.time()
        frame_rate = round(1 / (current_time - self.last_frame_time))
        self.last_frame_time = current_time
        cv2.putText(frame, "TENNISv1.0", TITLE_ORG, cv2.FONT_HERSHEY_SIMPLEX, TITLE_SCALE, TITLE_COLOR, TITLE_THICKNESS)
        cv2.putText(frame, f"Score: {self.score_player}", SCORE_ORG, cv2.FONT_HERSHEY_SIMPLEX, SCORE_SCALE, SCORE_COLOR, SCORE_THICKNESS)
        cv2.putText(frame, f"FPS: {frame_rate}", FPS_ORG, cv2.FONT_HERSHEY_SIMPLEX, FPS_SCALE, FPS_COLOR, FPS_THICKNESS)
        cv2.putText(frame, f"Press H to retarget", HINT_1_ORG, cv2.FONT_HERSHEY_SIMPLEX, HINT_SCALE, HINT_COLOR, HINT_THICKNESS)
        cv2.putText(frame, f"Press J to show mask", HINT_2_ORG, cv2.FONT_HERSHEY_SIMPLEX, HINT_SCALE, HINT_COLOR, HINT_THICKNESS)
        cv2.putText(frame, f"Press Q to quit", HINT_3_ORG, cv2.FONT_HERSHEY_SIMPLEX, HINT_SCALE, HINT_COLOR, HINT_THICKNESS)
        if self.target_manager.is_target_set:
                cv2.putText(frame, f"[Valid] Target saved at {self.target_manager.target_saved_time}", LOG_VALID_ORG, cv2.FONT_HERSHEY_SIMPLEX, LOG_SCALE, LOG_VALID_COLOR, LOG_THICKNESS)
        else:
                cv2.putText(frame, f"[Invalid] No valid target detected. Retrying...", LOG_INVALID_ORG, cv2.FONT_HERSHEY_SIMPLEX, LOG_SCALE, LOG_INVALID_COLOR, LOG_THICKNESS)
        cv2.imshow("Video Detection", frame)

        return frame

    def _cleanup(self) -> None:
        self.cap.release()
        if self.video_writer:
            self.video_writer.release()
        cv2.destroyAllWindows()


# 主函数
def main():
    if len(sys.argv) < 2:
        print("Usage: python3 main.py <video_path or image_path or 0 for stream> [output_path (optional)]")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = None

    if len(sys.argv) > 2:
        output_path = sys.argv[2]

    if input_path.endswith(('.mp4', '.avi', '.mov')):
        if output_path:
            processor = VideoProcessor(input_source=input_path, output_path=output_path)
        else:
            processor = VideoProcessor(input_source=input_path)
        processor.process_stream()
    elif input_path.endswith(('.jpg', '.png', '.jpeg')):
        print("Image processing not implemented yet")
    elif input_path == "0":
        processor = VideoProcessor(input_source=0)
        processor.process_stream()
    else:
        print(f"Error: Unsupported file type or invalid input {input_path}")
        sys.exit(1)


if __name__ == "__main__":
    main()
