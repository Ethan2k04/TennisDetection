import time
import json
import cv2
import os
from typing import Dict,Tuple
from pprint import pprint
from dataclasses import dataclass
import numpy as np
import math
import uuid
import urllib.parse
from pprint import pprint
from tools import  log_with_timestamp
from constants import (
    SETTINGS_FILE,X_COOR_WEIGHT, Y_COOR_WEIGHT, CONFIG_FILE,TARGET_COLOR,LINE_THICKNESS,TARGET_ROI_MARGIN,TEXT_MARGIN,FONT_SCALE,
    MIN_POLY, PERI_BIAS
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

        if os.path.exists(SETTINGS_FILE):
            with open(SETTINGS_FILE, 'r') as file:
                settings = json.load(file)
                self.target_scores = settings["target_score"]
                self.num_target = len(self.target_scores)
        else:
            self.num_target = 0

         # hit sum up score 
        self.score_player  = 0

        #for hit the score target, and push to server need data
        self.hit_x: int = 0
        self.hit_y : int = 0
        self.hit_score:int = 0
        self.device_id = urllib.parse.quote(TargetManager.get_mac_address())
        self.hit_target_id = 1


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
        target_result = self._detect_target(frame)
        self.last_relocate_time = time.time()
        # print(target_result)

        if self._is_target_result_valid(target_result, self.num_target):
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
                #push to server
                self.hit_score = score 
                self.hit_x = camera_point[0]
                self.hit_y = camera_point[1]
                self.hit_target_id = 1

                return True   
            
            target_white_circle = self.targets[5]
            score = target_white_circle.hit_white_target_circle_score_test(camera_point)
            if score > 0 :
                self.score_player += score
                # push to server 
                self.hit_score = score 
                self.hit_x = camera_point[0]
                self.hit_y = camera_point[1]
                self.hit_target_id = 5
                return True   
        else :    
            for key, target_circle in self.targets.items():
                score = target_circle.hit_score_test(camera_point)
                if score > 0 :
                    self.score_player += score
                     # push to server 
                    self.hit_score = score 
                    self.hit_x = camera_point[0]
                    self.hit_y = camera_point[1]
                    self.hit_target_id = key
                    return True    

        return False
    
    def get_push_score_data(self)->Dict:
        score_data = {
            "x": self.hit_x,
            "y": self.hit_y,
            "score": self.hit_score,
            "device_id": self.device_id,
            "target_id": str(self.hit_target_id),
            }
        return score_data


        
    # 检测目标轮廓
    def _detect_target(self,frame):
        """
         检测 frame 中的椭圆轮廓，并选择面积最大的 6 个椭圆作为目标。
         """
        # 转换为灰度图像
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 高斯模糊降噪
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # 边缘检测
        edges = cv2.Canny(blurred, 50, 150)

        # 查找轮廓
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 用于存储检测到的轮廓及其面积
        detected_contours = [contour for contour in contours if self._is_circle(contour) and cv2.contourArea(contour) > 100]

        # 如果有检测到的轮廓，选择面积最大的 TARGET_NUM 个作为结果类
        result_contours = []
        if len(detected_contours) > 0:
        # 将检测到的轮廓按面积从大到小排序
            sorted_contours = sorted(detected_contours, key=lambda c: cv2.contourArea(c), reverse=True)

          # 选择面积最大的 TARGET_NUM 个轮廓作为结果类
            if len(detected_contours) > self.num_target:
                result_contours = sorted_contours[:self.num_target]
            else:
                result_contours = sorted_contours[:len(detected_contours)]

        # 返回结果
        result = {score: [] for score in self.target_scores}

        for i in range(len(result_contours)):
            score = self.target_scores[i]
            result[score].append(result_contours[i])

        return result
    

    # 判断轮廓是否为圆形
    def _is_circle(self,contour):
        """
        根据轮廓点判断轮廓是否为类圆形。
        """
        if len(contour) < MIN_POLY:
            return False

         # 拟合椭圆
        (x, y), (major_axis, minor_axis), angle = cv2.fitEllipse(contour)

        # 计算圆度（长轴和短轴的比例）
        circularity = minor_axis / major_axis if major_axis != 0 else 0

        # 圆度接近1且面积大于一定阈值
        return circularity > 1.0 - PERI_BIAS and cv2.contourArea(contour) > 100

    # 判断目标结果集合是否符合设定
    def _is_target_result_valid(self, target_result, num_target):
        """
         判断靶标识别结果是否符合 settings 中的设定。
        """
        # 统计所有轮廓的总数
        total_contours = sum(len(contours) for contours in target_result.values())

        # 判断总数是否等于 num_target
        return total_contours == num_target

    # check target circle position order 
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
    
    @staticmethod
    def get_mac_address()->str:
        # 获取香橙派设备的MAC地址
        mac = uuid.getnode()
        mac_address = ':'.join(f'{(mac >> ele) & 0xff:02x}' for ele in range(40, -1, -8))
        return mac_address
