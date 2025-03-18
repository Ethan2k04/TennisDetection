import cv2 
import numpy as np
import math
from typing import List,Tuple

# -----------------------------------------------
# record tennis ball info
class TenisBall:
    def __init__(self, center_x: int, center_y: int, width: int, height: int, area: int, step_count:int):
        self.center_x = center_x
        self.center_y = center_y
        self.width = width
        self.height = height
        self.area = area
        self.step_count = step_count
        self.v_x = 0.0 
        self.v_y = 0.0 
        self.v_dot = 0.0

    def __repr__(self):
        return f"TenisBall(centerx={self.center_x}, centery={self.center_y}, width={self.width}, height={self.height}, area={self.area} ,step_count={self.step_count})"

    def get_center(self):
        return (self.center_x, self.center_y)

    def get_dimensions(self):
        return (self.width, self.height)

    def get_area(self):
        return self.area
    
    def get_step_count(self):
        return self.step_count

    def calculate_v(self, prev_ball: 'TenisBall'):
        self.v_x = (self.center_x - prev_ball.center_x) 
        self.v_y = (self.center_y - prev_ball.center_y) 
        # normalize
        mag = math.sqrt(self.v_x * self.v_x + self.v_y * self.v_y)
        self.v_x = self.v_x / mag
        self.v_y = self.v_y / mag 
        # self.a_x = self.v_x - prev_ball.v_x
        # self.a_y = self.v_y - prev_ball.v_y
        # not usefull in jude direction change
        #self.v_cross =  self.v_y * prev_ball.v_x - self.v_x * prev_ball.v_y
        self.v_dot = self.v_x * prev_ball.v_x + self.v_y * prev_ball.v_y

    def show_v(self):
        a = math.sqrt( self.a_x * self.a_x + self.a_y * self.a_y)
        return f"TenisBall(centerx={self.center_x}, centery={self.center_y}, v.x = {self.v_x}, v.y = {self.v_y}"

    def show_v_dot(self):
        return f"TenisBall(centerx={self.center_x}, centery={self.center_y}, v.x = {self.v_x}, v.y = {self.v_y}, v_dot={self.v_dot}"


# ---------------------------------------------------------------

class TennisBallManager:
    minCount = 6
    maxCount = 400
    maxStepCount = 30
    maxBallLastStepCount = 5

    def __init__(self):
        self.tennis_ball_path : List[TenisBall] = []
        self.last_ball_step_count = 0   # ball in target roi  no morth step count 20
        self.is_hit = False  # if hit , do not do hit test for following find ball

    def find_ball(self, roi:np.ndarray, step_count:int)->bool:
        green_binary = self.get_green_from_roi(roi)
        count = np.count_nonzero(green_binary)
        if count < TennisBallManager.minCount or count > TennisBallManager.maxCount:
            return False
        
        # the tennis ball appear in target region time 
        if step_count - self.last_ball_step_count > TennisBallManager.maxStepCount:
            print("******* clear  tennis ball path *********")
            self.tennis_ball_path.clear()
            self.is_hit = False

        # found ball 
        # print(f" find ball : {count}")
        self.last_ball_step_count = step_count
        rows, columns = np.where(green_binary > 0)
        center_x = int(np.mean(columns))
        center_y = int(np.mean(rows))
        
        max_x = int(np.max(columns))
        min_x = int(np.min(columns))
        width = max_x - min_x

        max_y = int(np.max(rows))
        min_y = int(np.min(rows))
        height = max_y - min_y
        ball = TenisBall(center_x=center_x,center_y= center_y,width=width,height=height,area=count,step_count= step_count)
        if len(self.tennis_ball_path) > 0:
            prev_ball = self.tennis_ball_path[-1]
            ball.calculate_v(prev_ball=prev_ball)

        self.tennis_ball_path.append(ball)

        # print(ball)
        print(ball.show_v_dot())
     
        return True
    


    def get_last_ball(self)->Tuple[int,int,int,int]:
        ball = self.tennis_ball_path[-1]
        return (ball.center_x, ball.center_y, ball.width, ball.height)

    def hit_test(self, step_count)->Tuple[int,int,bool] | None:
        if self.is_hit:
            return None
        
        if len(self.tennis_ball_path) == 0 :
            return None
        
        # exceed  maxBallLastStepCount frame no ball found, return last ball as hit 
        if (step_count - self.last_ball_step_count) > TennisBallManager.maxBallLastStepCount:
            last_ball = self.tennis_ball_path[-1]
            x = last_ball.center_x
            y = last_ball.center_y
            self.is_hit = True
            print("hit  exceed  maxBallLastStepCount")
            return (x,y, True)
        
        if len(self.tennis_ball_path) < 2:
            return None  # less to detect hit 
        
        last_ball_1 = self.tennis_ball_path[-1]
        last_ball_2 = self.tennis_ball_path[-2]
        if (last_ball_1.step_count - last_ball_2.step_count) > 2:
            x = last_ball_2.center_x
            y = last_ball_2.center_y
            self.is_hit = True
            print("hit white, discontinure frame found ball , fly in white and fly out white")
            return (x,y,True)
        
        # only 3 balls position ->  two velocity -> direction change 
        if len(self.tennis_ball_path) > 2:
            if last_ball_1.v_dot < 0.5:
                # direction change 
                x = last_ball_2.center_x
                y = last_ball_2.center_y
                self.is_hit = True
                print("hit direction change")
                return (x,y,False)
            
        return None
        

    def get_green_from_roi(self, roi: np.ndarray) -> np.ndarray:

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        # gray1 = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        # return gray1
        # 定义绿色在 HSV 颜色空间中的范围
        lower_green = np.array([30, 100, 100])
        upper_green = np.array([80, 255, 255])

        # 使用 cv2.inRange 函数创建一个二值图像，其中绿色像素为白色，其他为黑色
        mask = cv2.inRange(hsv, lower_green, upper_green)

        filtered_image = cv2.medianBlur(mask, 3)

        kernel = np.ones((3,3), np.uint8)
        opened_bitmap = cv2.morphologyEx(filtered_image, cv2.MORPH_OPEN, kernel)
        return  opened_bitmap



   
 

  
