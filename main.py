import sys
import time
import json
import cv2
import threading
from network import push_data
from kernel import detect_balls, detect_target, is_target_result_valid, detect_collision, draw_target_boxes, \
    draw_ball_boxes
from tools import create_trackbar, save_target_to_config
from constants import CONFIG_FILE, SCORE_ORG, SCORE_SCALE, SCORE_COLOR, SCORE_THICKNESS, FPS_SCALE, FPS_COLOR, \
    FPS_THICKNESS, RETARGET_WAIT_SEC, MAX_RETRY, RETRY_INTERVAL


# 目标管理类
class TargetManager:
    def __init__(self, num_target: int):
        self.is_target_set = False
        self.last_relocate_time = time.time()
        self.num_target = num_target
        self.target_data = {}

    def relocate_target(self, frame, retarget_wait_sec: float) -> None:
        if not self.is_target_set and time.time() - self.last_relocate_time > retarget_wait_sec:
            target_result = detect_target(frame)
            if is_target_result_valid(target_result, self.num_target):
                self.target_data = self._parse_target_result(target_result)
                save_target_to_config(self.target_data)
                print(f"[Valid] Target saved at {time.strftime('%Y-%m-%d %H:%M:%S')}")
                self.is_target_set = True
            else:
                print(f"[Invalid] No valid target detected. Retrying...")
            self.last_relocate_time = time.time()

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
    def __init__(self, input_source=0):
        self.cap = cv2.VideoCapture(input_source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Unable to access input source: {input_source}")

        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.target_manager = TargetManager(num_target=3)
        self.score_player = 0
        self.last_frame_time = time.time()
        self.ball_timestamps = {}

    def process_stream(self) -> None:
        create_trackbar()
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("End of video or failed to grab frame")
                break

            self.target_manager.relocate_target(frame, retarget_wait_sec=RETARGET_WAIT_SEC)
            frame = self._update_score(frame)

            self._display_frame(frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        self._cleanup()

    def _update_score(self, frame) -> cv2.Mat:
        ball_result = detect_balls(frame)
        with open(CONFIG_FILE, 'r') as file:
            config = json.load(file)
        frame = draw_ball_boxes(frame, ball_result)
        frame = draw_target_boxes(frame, config)
        for ball_id, ball in enumerate(ball_result):
            ball_center = (int((ball[0] + ball[2]) / 2), int((ball[1] + ball[3]) / 2))
            is_collided, score = detect_collision(ball_center, config)
            if is_collided:
                self.score_player += score
                score_data = {
                    "x": ball_center[0],
                    "y": ball_center[1],
                    "score": score
                }

                # 使用线程执行 push_data
                push_thread = threading.Thread(
                    target=push_data,
                    args=(score_data, MAX_RETRY, RETRY_INTERVAL)
                )
                push_thread.start()

        return frame

    def _display_frame(self, frame) -> None:
        current_time = time.time()
        frame_rate = round(1 / (current_time - self.last_frame_time))
        self.last_frame_time = current_time
        cv2.putText(frame, f"Score: {self.score_player}", SCORE_ORG, cv2.FONT_HERSHEY_SIMPLEX, SCORE_SCALE, SCORE_COLOR,
                    SCORE_THICKNESS)
        cv2.putText(frame, f"FPS: {frame_rate}", (self.frame_width - 160, 30), cv2.FONT_HERSHEY_SIMPLEX, FPS_SCALE,
                    FPS_COLOR, FPS_THICKNESS)
        cv2.imshow("Video Detection", frame)

    def _cleanup(self) -> None:
        self.cap.release()
        cv2.destroyAllWindows()


# 主函数
def main():
    if len(sys.argv) < 2:
        print("Usage: python3 main.py <video_path or image_path or 0 for stream>")
        sys.exit(1)

    input_path = sys.argv[1]
    if input_path.endswith(('.mp4', '.avi', '.mov')):
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
