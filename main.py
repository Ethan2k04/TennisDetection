import sys
import time
import json
import cv2
import os
import threading
from typing import Any
from network import push_data
from kernel import detect_balls, detect_target, is_target_result_valid, build_target_status, update_target_status, \
    draw_target_boxes, draw_ball_boxes
from tools import create_trackbar, save_target_to_config,  get_trackbar_reset_target_switch, log_with_timestamp
from constants import CONFIG_FILE, SCORE_ORG, SCORE_SCALE, SCORE_COLOR, SCORE_THICKNESS, FPS_SCALE, FPS_COLOR, \
    FPS_THICKNESS, RETARGET_WAIT_SEC, MAX_RETRY, RETRY_INTERVAL, SETTINGS_FILE, BALL_HIT_WAIT_SEC


# 目标管理类
class TargetManager:
    def __init__(self):
        self.is_target_set = False
        self.last_relocate_time = time.time()
        self.num_target = 0
        self.target_data = {}
        if os.path.exists(SETTINGS_FILE):
            with open(SETTINGS_FILE, 'r') as file:
                settings = json.load(file)
                self.num_target = settings["num_target"]

    def relocate_target(self, frame, retarget_wait_sec: float) -> bool:
        force_retarget = get_trackbar_reset_target_switch()
        if (not self.is_target_set or force_retarget) and time.time() - self.last_relocate_time > retarget_wait_sec:
            target_result = detect_target(frame)
            self.last_relocate_time = time.time()
            if is_target_result_valid(target_result, self.num_target):
                self.target_data = self._parse_target_result(target_result)
                save_target_to_config(self.target_data)
                log_with_timestamp(f"\033[92m[Valid] Target saved at {time.strftime('%Y-%m-%d %H:%M:%S')}\033[0m")
                self.is_target_set = True
                return True
            else:
                self.target_data = self._parse_target_result(target_result)
                save_target_to_config(self.target_data)
                log_with_timestamp(f"\033[93m[Invalid] No valid target detected. Retrying...\033[0m")
                self.is_target_set = False
                return False

    @staticmethod
    def _parse_target_result(target_result: dict) -> dict:
        target_data = {}
        target_id = 0
        for score, contours in target_result.items():
            for contour in contours:
                (x, y), (major_axis, minor_axis), angle = cv2.fitEllipse(contour)
                target_data[f"{target_id}"] = {
                    "cls": score,
                    "center_x": x,
                    "center_y": y,
                    "major_axis": major_axis,
                    "minor_axis": minor_axis,
                    "angle": angle,
                }
                target_id += 1
        return target_data


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

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        self._cleanup()

    def _update_score(self, frame) -> cv2.Mat:
        ball_result = detect_balls(frame)
        with open(CONFIG_FILE, 'r') as file:
            config = json.load(file)

        frame = draw_ball_boxes(frame, ball_result)
        frame = draw_target_boxes(frame, config)

        if self.target_status is not None:
            for ball_id, ball in enumerate(ball_result):
                ball_center = (int((ball[0] + ball[2]) / 2), int((ball[1] + ball[3]) / 2))
                collision_detected, score = update_target_status(self.target_status, ball_center, frame)
                if collision_detected and abs(time.time() - self.last_collision_time > BALL_HIT_WAIT_SEC):
                    if score != 0:
                        self.last_collision_time = time.time()
                        self.score_player += score
                        score_data = {
                            "x": ball_center[0],
                            "y": ball_center[1],
                            "score": score
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
        cv2.putText(frame, f"Score: {self.score_player}", SCORE_ORG, cv2.FONT_HERSHEY_SIMPLEX, SCORE_SCALE, SCORE_COLOR,
                    SCORE_THICKNESS)
        cv2.putText(frame, f"FPS: {frame_rate}", (self.frame_width - 160, 30), cv2.FONT_HERSHEY_SIMPLEX, FPS_SCALE,
                    FPS_COLOR, FPS_THICKNESS)
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
