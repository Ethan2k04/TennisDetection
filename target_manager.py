import time
import json
import cv2
import os
from typing import Dict,Tuple
from pprint import pprint
from dataclasses import dataclass
import numpy as np
import math
from pprint import pprint

from kernel import (
    detect_target, is_target_result_valid, build_target_status,
)

from tools import save_target_to_config, log_with_timestamp


from constants import (
    SETTINGS_FILE,X_COOR_WEIGHT, Y_COOR_WEIGHT, CONFIG_FILE,TARGET_COLOR,LINE_THICKNESS,TARGET_ROI_MARGIN,TEXT_MARGIN,FONT_SCALE
)

# ----------------------------------------------------------
# 目标区域
@dataclass
class TargetROI:
    left : int = 0 
    top : int = 0
    right : int = 0 
    bottom : int = 0

    def get_roi(self, frame):
        return frame[self.top:self.bottom, self.left:self.right].copy()
    
    def get_top_left(self):
        return (self.left,self.top)
    
    def get_bottom_right(self):
        return (self.right,self.bottom)
    
    def to_camera(self, point:Tuple[int,int])->Tuple[int,int]:
    # change relative point to camera point 
        x,y = point 
        return (x + self.left, y + self.top )
    
# ------------------------------------------------------

# 目标圆
class TargetCircle:
    def __init__(self, score :int , center_x:float, center_y:float, major_axis:float, minor_axis:float, angle:float):
        # for save /load
        self.score =  score
        self.center_x = center_x
        self.center_y = center_y
        self.major_axis = major_axis
        self.minor_axis = minor_axis
        self.angle = angle

        # for hit test ,roi , validation
        self.width = int(minor_axis)
        self.height = int(major_axis)
        self.left = int(center_x) - self.width // 2
        self.top = int(center_y) - self.height // 2
        self.right = int(center_x) + self.width // 2
        self.bottom = int(center_y) + self.height // 2
        self.radius = self.width // 2
        self.x = int(center_x)
        self.y = int(center_y)
    
    def to_save_dict(self):
            return {'cls': self.score,
                    'center_x': self.center_x,
                     'center_y':self.center_y,
                    'major_axis': self.major_axis,
                    'minor_axis': self.minor_axis,
                    'angle': self.angle
                    } 
    
    # point is global cooridate
    def hit_score_test(self, point:Tuple[int,int])-> int:
        '''
        is in circle return score 
        else return 0
        '''
        if self._is_in_rectangle(point=point):
            return self.score
        else :
            return 0
        
     # white circle special test. ball fly over white can not recognize
    def hit_white_target_circle_score_test(self,point:Tuple[int,int])->int :
        if self._is_near_circle(point=point):
            return self.score
        else :
            return 0

    def _is_in_rectangle(self, point:Tuple[int,int])->bool :
        x, y = point
        return x >= self.left  and x  <= self.right and y >= self.top and y <= self.bottom
             
    def _is_in_circle(self, point:Tuple[int,int])->bool :
        p_x,p_y = point
        return ((p_x - self.x) ** 2 + (p_y - self.y) ** 2) <= (self.height / 2) ** 2

    def _is_near_circle(self, point:Tuple[int,int])->bool :
        p_x,p_y = point
        # d = math.sqrt(((p_x - self.x) ** 2 + (p_y - self.y) ** 2))
        # print(f"distance:{d}    {self.height}")
        return ((p_x - self.x) ** 2 + (p_y - self.y) ** 2) <= self.height  ** 2
    
    def __str__(self):
        return f'''
            cls: {self.cls},
            center_x: {self.center_x},
            center_y: {self.center_y},
            major_axis: {self.major_axis},
            minor_axis: {self.minor_axis},
            angle: {self.angle}
        '''

# -----------------------------------------------------------------------
# 目标管理类
'''
1:10, 2:10, 3:30, 4:20, 5:20, 6: 30
'''
class TargetManager:
    def __init__(self):
        self.num_target = 0  # calc from settings
        self.target_scores : list[int] = []  #get from settings

        self.targets : Dict[int : TargetCircle ] = {}
        self.target_roi = TargetROI()
        self.is_target_set = False
        self.force_retarget = False
        self.last_relocate_time = time.time()
        self.target_saved_time = time.strftime('%Y-%m-%d %H:%M:%S')

        # hit sum up score 
        self.score_player  = 0

        if os.path.exists(SETTINGS_FILE):
            with open(SETTINGS_FILE, 'r') as file:
                settings = json.load(file)
                self.target_scores = settings["target_score"]
                self.num_target = len(self.target_scores)
        else:
            self.num_target = 0


    def calc_target_region(self):
        # find top
        top = min(self.targets[1].top,self.targets[2].top) - TARGET_ROI_MARGIN
        # find left
        left = min(self.targets[1].left,self.targets[3].left) - TARGET_ROI_MARGIN
        # find right
        right = max(self.targets[2].right,self.targets[6].right) + TARGET_ROI_MARGIN
        # find bottom
        bottom = max(self.targets[3].bottom,self.targets[4].bottom,self.targets[5].bottom,self.targets[6].bottom) + TARGET_ROI_MARGIN 
        self.target_roi = TargetROI(left=left,top=top,right=right,bottom=bottom) 

    # 绘制标靶目标框,包括6个目标
    def draw_target_region(self,frame):
        cv2.rectangle(frame, self.target_roi.get_top_left(),self.target_roi.get_bottom_right(),TARGET_COLOR,LINE_THICKNESS)
        return frame
    
    # 根据检测结果绘制标靶目标框
    def draw_target_circles(self,frame):
        """
        根据检出的标靶轮廓绘制目标框和椭圆。
        """
        for key, target_circle in self.targets.items():
            score = target_circle.score  # 甲方说要显示序号而不是分数（保留）
            x = int(target_circle.center_x)  # 椭圆中心点 x 坐标
            y = int(target_circle.center_y)  # 椭圆中心点 y 坐标
            major_axis = int(target_circle.major_axis)  # 椭圆的长轴长度
            minor_axis = int(target_circle.minor_axis)  # 椭圆的短轴长度
            # label = "Target_" + str(key)  # 标签
            label =  str(key)  # 标签
            text_position = (x + int(minor_axis / 2) - TEXT_MARGIN, y - int(major_axis / 2) + TEXT_MARGIN)

            # 绘制椭圆
            cv2.ellipse(frame, (x, y), (minor_axis // 2, major_axis // 2), 0, 0, 360, TARGET_COLOR, LINE_THICKNESS)

             # 在框旁边显示标签
            cv2.putText(
             frame, label, text_position, cv2.FONT_HERSHEY_SIMPLEX,
            FONT_SCALE, TARGET_COLOR, LINE_THICKNESS
            )

        return frame
    
    # center_x, center_y is relative to roi
    def draw_ball_in_roi(self, frame , center_x, center_y, width, height):
        y1 = self.target_roi.top + center_y - height // 2
        x1 = self.target_roi.left + center_x - width // 2
        y2 = y1 + height
        x2 = x1 + width  

        cv2.rectangle(frame, (x1, y1), (x2, y2), TARGET_COLOR, LINE_THICKNESS)

        return frame


    def load_config(self):
        with open(CONFIG_FILE, 'r') as file:
            config = json.load(file)
            for k, v in config.items :
                targetCircle = TargetCircle(v['cls'], v['center_x'], v['center_y'], v['major_axis'], v['minor_axis'],v['angle'])
                self.targets[k] = targetCircle

        pprint(self.targets)


    # 定义保存目标框信息的函数
    def save_config(self):
        targets_dict =  { key: targetCircle.to_save_dict()  for key , targetCircle in self.targets.items() }
        config = {"target_saved_time" : self.target_saved_time}
        # 将新的目标框信息添加到配置文件中
        config.update(targets_dict)

        # pprint(config)
   
        # 保存更新后的信息
        with open(CONFIG_FILE, 'w') as file:
            json.dump(config, file, indent=4)


    def relocate_target(self, frame, retarget_wait_sec: float) -> bool:
        if self.is_target_set and (not self.force_retarget):
            return True # 不需要设置
        
        # if time.time() - self.last_relocate_time < retarget_wait_sec:
        #     return False # 需要设置，离上次设置时间太短
               
	    # 亮度修正
        frame = self._adjust_brightness(frame)
        target_result = detect_target(frame)
        self.last_relocate_time = time.time()
        # print(target_result)

        if is_target_result_valid(target_result, self.num_target):
            target_data = self._parse_target_result(target_result)
            targets_dict = self._validate_target_result(target_data)
            if targets_dict == {}:
                log_with_timestamp(f"\033[92m[InValid] Target \033[0m")
                return False
            
            self.targets = targets_dict
            self.calc_target_region() 
            self.save_config()
            self.target_saved_time = time.strftime('%Y-%m-%d %H:%M:%S')
            log_with_timestamp(
                 f"\033[92m[Valid] Target saved at {self.target_saved_time}\033[0m"
            )
            self.is_target_set = True
            self.force_retarget = False

            return True
        else:
            log_with_timestamp(
                    "\033[93m[Invalid] No valid target detected. Retrying...\033[0m"
                )

            return False
        
        
    def hit_score_test(self,point,is_near_white)->bool:
        """
        return true : hit target and add score 
        False not hit score 
        """
        # chagne roi coordinate to camera global corrdinate 
        camera_point = self.target_roi.to_camera(point)
        # pprint(self.targets.values())
        if is_near_white : 
            print(" hit in white circle check")
            target_white_circle = self.targets[1]
            score = target_white_circle.hit_white_target_circle_score_test(camera_point)
            if score > 0 :
                self.score_player += score
                return True   
            
            target_white_circle = self.targets[5]
            score = target_white_circle.hit_white_target_circle_score_test(camera_point)
            if score > 0 :
                self.score_player += score
                return True   
        else :    
            for target_circle in self.targets.values():
                score = target_circle.hit_score_test(camera_point)
                if score > 0 :
                    self.score_player += score
                    return True    

        return False

        
    def _validate_target_result(self,target_data)-> Dict[int,TargetCircle]:
        targets = {}
        for k, v in target_data.items() :
            key = int(k)
            score = v['cls']
            center_x = v['center_x']
            center_y = v['center_y']
            major_axis = v['major_axis']
            minor_axis = v['minor_axis']
            angle =  v['angle']
            target_circle = TargetCircle(score=score,center_x=center_x,center_y=center_y,major_axis=major_axis,minor_axis=minor_axis,angle=angle)
            targets[key] = target_circle

        """
            space position:
            1       2
            3  4  5  6
        """
        # check target 1
        if targets[1].right > targets[2].left:
            return {}
        
        top_3456 = min(targets[3].top,targets[4].top,targets[5].top,targets[6].top)

        if targets[1].bottom > top_3456:
            return {}
        
        # check target 2
        if targets[2].bottom > top_3456:
            return {}
        
        #chek target 3
        if targets[3].right > targets[4].left:
            return {}
        
        #check target 4 
        if targets[4].right > targets[5].left:
            return {}
        
        #check target 5
        if targets[5].right > targets[6].left:
            return {}
        
        # pass the space position check 
        return targets
        


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
    
    def _adjust_brightness(self,frame):
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge((l, a, b))
        frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        return frame
