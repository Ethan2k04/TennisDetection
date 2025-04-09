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
from scipy.signal import find_peaks
from pprint import pprint
from tools import  log_with_timestamp
from constants import (
    SETTINGS_FILE,X_COOR_WEIGHT, Y_COOR_WEIGHT, CONFIG_FILE,LINE_THICKNESS,TARGET_ROI_MARGIN,TEXT_MARGIN,FONT_SCALE,
    MIN_POLY, PERI_BIAS,BALL_COLOR,HIT_COLOR
)
from tennis_ball_manager import TennisBallManager, TenisBall, TenisBallInfo
#-----------------------------------------------------------



# ----------------------------------------------------------
# 目标区域
@dataclass
class TargetROI:
    # global cordinator, 
    left : int = 0 
    top : int = 0
    right : int = 0 
    bottom : int = 0

    #current_target_roi
    height : int = 0
    width  : int = 0
    scope : int = 0
    is_target_roi = False

    #background for black score 10
    # relative to self
    center_x : int = 0 
    center_y : int = 0 
    radius : int = 0
    inner_radius: int = 0

    # for improve performance
    # first time or press H, to relocate the target roi
    # roi  histogram info about hue,saturation, value
    is_need_caculate_bg_hsv = True

    # max peak , white , black  hue is uncertain 
    hue : int = 0 
    # low end peak, for white saturation 
    white_saturation: int = 0 
    # max peak, for target black value / brightness
    black_value :int = 0

    # frame is from videocapture
    def get_roi(self, frame):
        return frame[self.top:self.bottom, self.left:self.right].copy()
    
    #input center is relaive to target_roi
    def get_current_target_roi(self,center:Tuple[int,int]|None):
        if center is None :
            self.is_target_roi = True
            return self
        
        x,y = center
        left = max(0,x - self.scope) + self.left
        top =  max(0, y - self.scope) + self.top
        right = min(self.width, x + self.scope) + self.left
        bottom = min(self.height , y + self.scope) + self.top

        return TargetROI(left=left,top=top,right=right,bottom=bottom)
        

    def get_top_left(self):
        return (self.left,self.top)
    
    def get_bottom_right(self):
        return (self.right,self.bottom)
    
    def to_camera(self, point:Tuple[int,int])->Tuple[int,int]:
    # change relative point to camera point 
        x,y = point 
        return (x + self.left, y + self.top )
    
    #  first time or trigger by press H
    def setup_for_black_score_10(self):
        w = self.right - self.left
        h = self.bottom - self.top
        self.center_x = w // 2
        self.center_y = h // 2
        self.radius = min(w,h) // 2
        self.inner_radius = self.radius * 0.6

    def setup_scope(self):
        self.width = self.right - self.left
        self.height = self.bottom - self.top
        self.scope = min(self.width,self.height) // 2

    def set_bg_hsv(self, h:int , s:int , v:int ):
        self.hue = h 
        self.white_saturation = s
        self.black_value = v
        self.is_need_caculate_bg_hsv = False

    def show_background_hsv(self):
        info = f'hue:{self.hue}  white staturation:{self.white_saturation} black value:{self.black_value}'
        print(info)

    # check point is in target circle 
    def is_in_circle(self, x:int, y:int)->bool:
        return (x - self.center_x) ** 2 + (y - self.center_y) ** 2 < self.radius ** 2
    
    # check point is in inner target circle
    def is_in_inner_circle(self, x:int, y:int)->bool:
        return (x - self.center_x) ** 2 + (y - self.center_y) ** 2 < self.inner_radius ** 2
    
    def get_binary_roi(self,frame:np.ndarray)->np.ndarray:
        pass
        # remove black, white ,  white sat , black value
        # remove background , hue histogram , peak,  bin 90
        # left is ball or no ball 
    



    
    
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
        self.left_ = self.left - 5

        self.top = int(center_y) - self.height // 2
        self.top_ = self.top - 5

        self.right = int(center_x) + self.width // 2
        self.right_ = self.right + 5

        self.bottom = int(center_y) + self.height // 2
        self.bottom_ = self.bottom + 5

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
        # print(f'x:{x} y:{y} left: {self.left}  top: {self.top} right:{self.right} bottom:{self.bottom}')
        return   self.left_ <= x <= self.right_ and  self.top_ <= y <= self.bottom_
             
    def _is_in_circle(self, point:Tuple[int,int])->bool :
        p_x,p_y = point
        return ((p_x - self.x) ** 2 + (p_y - self.y) ** 2) <= (self.height / 2) ** 2

    # for white score check ,  point close to white score circle
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

        #包含幕布和五个目标的区域
        self.target_roi = TargetROI()

        # 由发现网球决定下帧搜索范围: for robust and fast
        self.current_target_roi = TargetROI()

        # init find ball parameters, use substract,  remove black , remove white, then left green ball
        self.target_roi_black_10 = TargetROI()
        

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


    # according 1,2,3,4,5,6 target circle , caulate target_roi, target_black_score_10 roi 
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
        self.target_roi.setup_scope()

        # black score 10 roi 
        top = self.targets[2].top
        left = self.targets[2].left
        right = self.targets[2].right
        bottom = self.targets[2].bottom
        self.target_roi_black_10 = TargetROI(left=left,top=top,right=right,bottom=bottom) 
        self.target_roi_black_10.setup_for_black_score_10()

    # center_point is relative to target_roi
    def setup_current_target_roi(self,center_point:Tuple[int,int]|None):
        self.current_target_roi = self.target_roi.get_current_target_roi(center=center_point)
        

    # 绘制标靶目标框,包括6个目标
    def draw_target_region(self,frame, target_color):
        cv2.rectangle(frame, self.target_roi.get_top_left(),self.target_roi.get_bottom_right(),target_color,LINE_THICKNESS)
        return frame
    
    def draw_current_target_roi(self,frame, target_color):
        cv2.rectangle(frame, self.current_target_roi.get_top_left(),self.current_target_roi.get_bottom_right(),target_color,8)
        return frame
    
    # 根据检测结果绘制标靶目标框
    def draw_target_circles(self,frame,target_color):
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
            cv2.ellipse(frame, (x, y), (minor_axis // 2, major_axis // 2), 0, 0, 360, target_color, LINE_THICKNESS)

             # 在框旁边显示标签
            cv2.putText(
             frame, label, text_position, cv2.FONT_HERSHEY_SIMPLEX,
            FONT_SCALE, target_color, LINE_THICKNESS
            )

        return frame
    
    # center_x, center_y is relative to roi
    def draw_ball_in_roi(self, frame , center_x, center_y, width, height):
        y1 = self.target_roi.top + center_y - height // 2
        x1 = self.target_roi.left + center_x - width // 2
        y2 = y1 + height
        x2 = x1 + width  

        cv2.rectangle(frame, (x1, y1), (x2, y2), BALL_COLOR , LINE_THICKNESS)

        return frame
    
    # center_x, center_y is relative to roi
    def draw_hit_in_roi(self, frame , center_x, center_y):
        y1 = self.target_roi.top + center_y 
        x1 = self.target_roi.left + center_x 
        radius = 40
        
        cv2.circle(frame, (x1, y1), radius, HIT_COLOR , -1)

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
        
    #when detect 2 ball like blob, should check only in target circle proority
    def check_ball_like_center_in_target_circle(self,point)->bool :
        """
        pint is relative to target roi
        return true : hit target 
        False not hit 
       
        """
        # print("check_ball_like_center_in_target_circle")
        # not the larget roi 
        if not self.current_target_roi.is_target_roi:
            # print("not in largest target roi")
            return False
        
        # chagne roi coordinate to camera global corrdinate 
        camera_point = self.target_roi.to_camera(point)

        # print(f"{camera_point}")
        # pprint(self.targets.values())
        # first check hit point is in target rectangle
        for key, target_circle in self.targets.items():
            score = target_circle.hit_score_test(camera_point)
            if score > 0 :
               return True
        
        # print('not in target circle')
        return False
    
            
    def hit_score_test(self,point,is_near_white)->bool:
        """
        pint is relative to target roi
        return true : hit target and add score 
        False not hit score 
        """
        # chagne roi coordinate to camera global corrdinate 
        camera_point = self.target_roi.to_camera(point)
        # pprint(self.targets.values())
        # first check hit point is in target rectangle
        for key, target_circle in self.targets.items():
            score = target_circle.hit_score_test(camera_point)
            if score > 0 :
                self.score_player += score
                # push to server data need
                self.hit_score = score 
                self.hit_x = camera_point[0]
                self.hit_y = camera_point[1]
                self.hit_target_id = key
                return True    
            
        #fly over white , loss ball
        if not is_near_white:
            return False
        
        # hit not in target circle
       
        # print(" hit in white circle check")
        #check white target 1 : score 10
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
        
        #check white target 5: score 30
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
       
    
    def get_push_score_data(self)->Dict:
        score_data = {
            "x": self.hit_x,
            "y": self.hit_y,
            "score": self.hit_score,
            "device_id": self.device_id,
            "target_id": str(self.hit_target_id),
            }
        return score_data
    
    # 2d -> 1d
    # black score 10 circle pixel to 1d  array  for caculate histograme
    def _convert_target_circle(self, roi:np.ndarray)->np.ndarray:
        list = []
        for (x, y), value in np.ndenumerate(roi):
            # if self.target_roi_black_10.is_in_circle(x,y):
            if self.target_roi_black_10.is_in_inner_circle(x,y):
                list.append(value)
        
        return np.array(list)



    # set target black score 10 background hsv
    def set_target_black_10_bg_hsv(self, roi: np.ndarray) :

        if not self.target_roi_black_10.is_need_caculate_bg_hsv:
            return
        
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        hue, saturation, value = cv2.split(hsv)

        # bin 180 -> bin 90
        hue = self._convert_target_circle(hue)
        hist = cv2.calcHist([hue], [0], None, [90], [0, 179])
        h = np.argmax( hist) * 2
        # print(f"hue:{h}")

        # bin 256 -> bin 128 
        saturation = self._convert_target_circle(saturation)
        # hist = cv2.calcHist([saturation], [0], None, [128], [0, 255])
        hist = cv2.calcHist([saturation], [0], None, [64], [0, 255])
        # 查找峰值
        peaks, _ = find_peaks(hist.flatten())

        # 打印峰值的索引和对应的计数值
        # for peak in peaks:
        #     print(f"峰值索引: {peak}, 计数值: {hist[peak]}")

        s = peaks[0] * 4
        # s = np.argmax( hist) * 2
        # print(f"saturation:{s}")

        # bin 256 -> bin 128
        value = self._convert_target_circle(value)
        hist = cv2.calcHist([value], [0], None, [128], [0, 255])
        v = np.argmax( hist) * 2
        # print(f"value:{v}")
        # print( "*" * 50)

        self.target_roi_black_10.set_bg_hsv(h,s,v )
        # self.target_roi_black_10.show_background_hsv()


    def get_ball_like_in_black_score_10(self, roi:np.ndarray)->np.ndarray:
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        binary_bitmap = np.ones_like(h) * 255

        # y is row , x is column
        for (x, y), value in np.ndenumerate(v):
            h_value = h[x, y]
            s_value = s[x,y]
            v_value = v[x,y]

            if not self.target_roi_black_10.is_in_circle(x,y):
                binary_bitmap[x,y] = 0
                continue

            # from histogram observe hue has a pike , green hue gap 
            # depend on  light : not reliable
            # if h_value > self.target_roi_black_10.hue - 20:
            #     binary_bitmap[x,y] = 0

            # remove black color : black h , s, v, only v is reliable, vq_value < peak 
            if v_value < self.target_roi_black_10.black_value + 60:
                binary_bitmap[x,y] = 0
                continue

        
            # remove white , use saturation , while saturatin is low 
            # if s_value < self.target_roi_black_10.white_saturation + 40 :
            #     binary_bitmap[x,y] = 0
            #     continue

            # green ball looklike
            if h_value > 75 or h_value < 35:
                binary_bitmap[x,y] = 0
                continue

        # return binary_bitmap

        # very in important in practise
        kernel = np.ones((5,5), np.uint8)
        # kernel = np.ones((3,3), np.uint8)
        opened_bitmap = cv2.morphologyEx(binary_bitmap, cv2.MORPH_OPEN, kernel)
        return opened_bitmap
    
    # according info, give a score , ball like 
    # only use in black score 10
    def _get_ball_like_score(self, left, top, width,height, area, centroid_x, centroid_y)->int :
        score = 0 
        if self.target_roi_black_10.is_in_circle(centroid_x,centroid_y):
            if 20 < area < 100:
                score += 10

            if 100 < area  < 500:
                score += 20

            if width * height < area * 2:
                score += 10

        # if centroid_x > left and centroid_x < left + width:
        #     score += 5
        # if centroid_y > top and centroid_y < top + height :
        #     score += 5

        # print(f'jude ball score:{score}')

        return score
    
    # input point is relative to black score 10 
    def _convert_black_score_10_to_target_roi( self, point:Tuple[int,int])->Tuple[int,int]:
        x = self.target_roi_black_10.left - self.target_roi.left + point[0]
        y = self.target_roi_black_10.top - self.target_roi.top + point[1]
        return (int(x),int(y))
    
    # input point is relative to current target roi
    # output point is relative to target roi 
    def _convert_current_target_roi_to_target_roi( self, point:Tuple[int,int])->Tuple[int,int]:
        rel_x, rel_y = point
        x = self.current_target_roi.left - self.target_roi.left + rel_x
        y = self.current_target_roi.top - self.target_roi.top + rel_y
        return (int(x),int(y))
      # 计算检测到球的颜色信息
    def _compute_ball_info(self, origin_image:np.ndarray, mask:np.ndarray)->TenisBallInfo:
        hsv = cv2.cvtColor(origin_image, cv2.COLOR_BGR2HSV)
        hue, saturation, value = cv2.split(hsv)
        mask_hue = hue[mask]
        h_mean = round(np.mean(mask_hue))
        h_median = round(np.median(mask_hue))
        h_std = round(np.std(mask_hue))
        # print(f'h mean :{h_mean} h_median: {h_median} h std : {h_std}')

        mask_saturation = saturation[mask]
        sat_mean = round(np.mean(mask_saturation))
        sat_median = round(np.median(mask_saturation))
        sat_std = round(np.std(mask_saturation))
        # print(f'sat_mean :{sat_mean} sat_median:{sat_median} sat_std : {sat_std}')

        mask_value = value[mask]
        v_mean = round(np.mean(mask_value))
        v_median = round(np.median(mask_value))
        v_std = round(np.std(mask_value))
        # print(f'v_mean :{v_mean}  v_median :{v_median } v_std : {v_std}')

        return TenisBallInfo(hue_mean=h_mean,hue_median=h_median,hue_std=h_std,
                    sat_mean=sat_mean,sat_median=sat_median,sat_std=sat_std,
                    value_mean=v_mean,value_median=v_median,value_std=v_std  )
    
    def find_ball_in_black_score_10(self, green_binary:np.ndarray, origin_image:np.ndarray,step_count:int)->Tuple[TenisBall|None,TenisBallInfo|None]:
      # binary green: background 0, forground 255
        area = np.count_nonzero(green_binary)
        # print(f'black score roi : ball area count:{area}')
        if area < TennisBallManager.minCount:
            return None,None
        
        # for print labels
        # np.set_printoptions(threshold=np.inf)

        best_left, best_top = 0, 0
        best_width, best_height = 0,0
        best_area , best_label = 0, 0
        best_centroid_x , best_centroid_y = 0,0 
        best_score = 0

        # 进行连通组件分析
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(green_binary, connectivity=8)
        # 检测到多个目标，不可靠，球撞击幕布
        if num_labels > 3:
            return None,None
        
        for i in range(1, num_labels):
            # print(f"Component {i}:")
            # print(f"  Left: {stats[i, 0]}")
            # print(f"  Top: {stats[i, 1]}")
            # print(f"  Width: {stats[i, 2]}")
            # print(f"  Height: {stats[i, 3]}")
            # print(f"  Area: {stats[i, 4]}")
            # print(f"  Centroid: ({centroids[i, 0]}, {centroids[i, 1]})")
            # print(labels)
            # print()
            left ,top = stats[i, 0], stats[i, 1]
            width, height = stats[i, 2],stats[i, 3]
            area , label = stats[i, 4], i
            centroid_x , centroid_y = centroids[i, 0],centroids[i, 1]
            score = self._get_ball_like_score(left=left,top=top,width=width,height=height,area=area,centroid_x=centroid_x,centroid_y=centroid_y)
            if score > best_score:
                best_score = score
                best_left ,best_top = left, top
                best_width, best_height = width,height
                best_area , best_label = area, label
                best_centroid_x , best_centroid_y = centroid_x,centroid_y

        # found tennis ball 
        if best_score > 0 : 
            x,y = self._convert_black_score_10_to_target_roi((best_centroid_x,best_centroid_y))
            ball = TenisBall(center_x=x, center_y=y,width= best_width,height=best_height,area= best_area,step_count=step_count)
            mask = labels == best_label
            ball_info = self._compute_ball_info(origin_image=origin_image,mask=mask)
            return ball,ball_info
        else:
            return None,None

    # for test and study 
    def get_hue_from_roi(self, roi: np.ndarray) -> np.ndarray:

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        hue, saturation, value = cv2.split(hsv)

        # 创建一个与色相通道形状相同的零数组
        hue_visual = np.zeros_like(hsv)
        # 将色相通道的值赋给新图像的所有通道
        hue_visual[:, :, 0] = hue
        hue_visual[:, :, 1] = hue
        hue_visual[:, :, 2] = hue

        return hue_visual
    
    # for test and study 
    def get_hue_hist_from_roi(self, roi: np.ndarray) -> np.ndarray:

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        hue, saturation, value = cv2.split(hsv)

        # only in circle is include 
        hue = self._convert_target_circle(hue)
        hist = cv2.calcHist([hue], [0], None, [90], [0, 179])

        return hist
    
     # for test and study 
    def get_s_from_roi(self, roi: np.ndarray) -> np.ndarray:

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        hue, saturation, value = cv2.split(hsv)

        # 创建一个与色相通道形状相同的零数组
        saturation_visual = np.zeros_like(hsv)
        # 将色相通道的值赋给新图像的所有通道
        saturation_visual[:, :, 0] = saturation
        saturation_visual[:, :, 1] = saturation
        saturation_visual[:, :, 2] = saturation

        return saturation_visual
    
     # for test and study 
    def get_s_hist_from_roi(self, roi: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        hue, saturation, value = cv2.split(hsv)

        saturation = self._convert_target_circle(saturation)
        hist = cv2.calcHist([saturation], [0], None, [128], [0, 255])
        return hist
    
    
    def get_v_from_roi(self, roi: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        hue, saturation, value = cv2.split(hsv)

        # 创建一个与色相通道形状相同的零数组
        value_visual = np.zeros_like(hsv)
        # 将色相通道的值赋给新图像的所有通道
        value_visual[:, :, 0] = value
        value_visual[:, :, 1] = value
        value_visual[:, :, 2] = value

        return value_visual
    
    def get_v_hist_from_roi(self, roi: np.ndarray) -> np.ndarray:

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        hue, saturation, value = cv2.split(hsv)

        value = self._convert_target_circle(value)
        hist = cv2.calcHist([value], [0], None, [128], [0, 255])
        return hist
        

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
    
    # relocate target 5 circle target need do histogram equilibrium 
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
        # when hit the store , push data to server need
        # 获取香橙派设备的MAC地址
        mac = uuid.getnode()
        mac_address = ':'.join(f'{(mac >> ele) & 0xff:02x}' for ele in range(40, -1, -8))
        return mac_address
