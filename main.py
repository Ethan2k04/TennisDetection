import sys
import time
import cv2
import threading

from network import push_data_worker,push_data_async
from tools import  log_with_timestamp
from target_manager import TargetManager
from tennis_ball_manager import TennisBallManager

from constants import (
    BALL_HIT_WAIT_SEC,  FPS_COLOR, FPS_SCALE, FPS_THICKNESS,
    HINT_COLOR, HINT_SCALE, HINT_THICKNESS, LOG_INVALID_COLOR, LOG_SCALE,
    LOG_THICKNESS, LOG_VALID_COLOR, RETARGET_WAIT_SEC, SCORE_COLOR, SCORE_SCALE, SCORE_THICKNESS,
    TITLE_COLOR, TITLE_SCALE, TITLE_THICKNESS,
    TITLE_ORG_RATIO, SCORE_ORG_RATIO, FPS_ORG_RATIO, HINT_1_ORG_RATIO,
    HINT_2_ORG_RATIO, LOG_INVALID_ORG_RATIO, LOG_VALID_ORG_RATIO,
    DEFAULT_FRAME_WIDTH, 
)

# -----------------------------------------------------
# 视频处理类
class VideoProcessor:
    def __init__(self, input_source=0, output_path=None):
        self.cap = cv2.VideoCapture(input_source)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        self.output_path = output_path

        self.target_manager = TargetManager()
        self.ball_manager = TennisBallManager()
      
        self.last_frame_time = time.time()
        self.ball_timestamps = {}
       
        self.last_collision_time = time.time()
        self.ball_hit_sec = BALL_HIT_WAIT_SEC
        self.retarget_sec = RETARGET_WAIT_SEC
     
        self.frame_counter = 0
        self.last_fps = 0
        self.last_fps_calc_time = time.time()

        self.step_count = 0 #very important, act as time 

        if not self.cap.isOpened():
            raise RuntimeError(f"Unable to access input source: {input_source}")
        
        if self.output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(
                self.output_path, fourcc, self.fps, (self.frame_width, self.frame_height)
            )
        else:
            self.video_writer = None

    def process_stream(self) -> None:
        while True:
            ret, frame = self.cap.read()
            self.ball_hit_sec, self.retarget_sec = (BALL_HIT_WAIT_SEC, RETARGET_WAIT_SEC)

            if not ret:
                log_with_timestamp("\033[93mEnd of video or failed to grab frame\033[0m")
                break

            self.step_count += 1
            # print('step_count = ', self.step_count)
            
            # find canvas target, it is important, if not find target circle , continue
            if not self.target_manager.relocate_target(frame, retarget_wait_sec=self.retarget_sec):
                continue
            
            # print("after target relocated")
            #find ball 
            #get target ROI , convert binary according tennis ball color
            target_roi = self.target_manager.target_roi.get_roi(frame=frame)
    
            # in target roi find ball 
            if self.ball_manager.find_ball(target_roi,self.step_count):
                x,y ,w, h = self.ball_manager.get_last_ball()
                frame = self.target_manager.draw_ball_in_roi(frame=frame,center_x=x, center_y =y,width= w,height= h)
    
            # ball hit canvas test 
            hit_result = self.ball_manager.hit_test(step_count= self.step_count)
            if hit_result:
                x,y, is_near_white = hit_result
                if self.target_manager.hit_score_test((x,y), is_near_white=is_near_white):
                    # show score in frame and send to server 
                    # print(f"********* scores: {target_manager.score_player} ********")
                    score_data = self.target_manager.get_push_score_data()
                    push_data_async(score_data)

            frame = self._display_frame(frame)

            if self.video_writer:
                self.video_writer.write(frame)

            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break
            elif key & 0xFF == ord('h'):
                self.target_manager.force_retarget = True

        self._cleanup()

 
    def _display_frame(self, frame) -> cv2.Mat:
        # 获取当前屏幕的宽度和高度
        frame_width = frame.shape[1]
        frame_height = frame.shape[0]
        frame_scale = frame_width / DEFAULT_FRAME_WIDTH
        # if frame is not None:
        #     frame_display = frame.copy()

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


        frame = self.target_manager.draw_target_region(frame=frame)
        frame = self.target_manager.draw_target_circles(frame=frame)

        # 绘制各种信息
        cv2.putText(
            frame, "TENNISv1.2", title_org, cv2.FONT_HERSHEY_SIMPLEX,
            TITLE_SCALE * frame_scale, TITLE_COLOR, int(TITLE_THICKNESS * frame_scale)
        )
        cv2.putText(
            frame, f"Score: {self.target_manager.score_player}", score_org, cv2.FONT_HERSHEY_SIMPLEX,
            SCORE_SCALE * frame_scale, SCORE_COLOR, int(SCORE_THICKNESS * frame_scale)
        )
        cv2.putText(
            frame, f"FPS: {frame_rate}", fps_org, cv2.FONT_HERSHEY_SIMPLEX,
            FPS_SCALE * frame_scale, FPS_COLOR, int(FPS_THICKNESS * frame_scale)
        )
        cv2.putText(
            frame, f"Press H to retarget", hint_1_org, cv2.FONT_HERSHEY_SIMPLEX,
            HINT_SCALE * frame_scale, HINT_COLOR, int(HINT_THICKNESS * frame_scale)
        )
        cv2.putText(
            frame, f"Press Q to quit", hint_2_org, cv2.FONT_HERSHEY_SIMPLEX,
            HINT_SCALE * frame_scale, HINT_COLOR, int(HINT_THICKNESS * frame_scale)
        )
        if self.target_manager.is_target_set:
            cv2.putText(
                frame, f"[Valid] Target saved at {self.target_manager.target_saved_time}",
                log_valid_org, cv2.FONT_HERSHEY_SIMPLEX, LOG_SCALE * frame_scale,
                LOG_VALID_COLOR, int(LOG_THICKNESS * frame_scale)
            )
        else:
            cv2.putText(
                frame, f"[Invalid] No valid target detected. Retrying...",
                log_invalid_org, cv2.FONT_HERSHEY_SIMPLEX, LOG_SCALE * frame_scale,
                LOG_INVALID_COLOR, int(LOG_THICKNESS * frame_scale)
            )

        cv2.imshow("Video Detection", frame)

        return frame

    def _cleanup(self) -> None:
        self.cap.release()
        if self.video_writer:
            self.video_writer.release()

        cv2.destroyAllWindows()


# 摄像头进程
def main():
    # 如果没有传递参数，则默认 input_path 为 "0"
    if len(sys.argv) < 2:
        input_path = "0"  # 默认使用摄像头
        output_path = None
    else:
        input_path = sys.argv[1]
        output_path = None
        if len(sys.argv) > 2:
            output_path = sys.argv[2]

    # thread is to push data to server
    push_data_thread = threading.Thread(target=push_data_worker)
    push_data_thread.daemon = True
    push_data_thread.start()

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
