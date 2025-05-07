import cv2 
import numpy as np
import math
from typing import List,Tuple
from dataclasses import dataclass
from hit_manager import HitCanvas


#--------------------------------------------
# 场景中网球的颜色信息，从检测到的网球提起
@dataclass
class TenisBallInfo:
    hue_mean : int = -1
    hue_median : int = 0
    hue_std : int = 0

    sat_mean : int = 0
    sat_median : int = 0
    sat_std : int = 0

    value_mean :int = 0
    value_median :int = 0
    value_std : int = 0

    def __str__(self):
        return f'h mean: {self.hue_mean}, hue media : {self.hue_median}'

    def __repr__(self):
        return f'h mean: {self.hue_mean}, hue media : {self.hue_median}'
    
    def show_info(self):
        print('*'*20)
        print(f'h mean:{self.hue_mean}, hu median:{self.hue_median} h std:{self.hue_std}')
        print(f'sat mean:{self.sat_mean}, sat median:{self.sat_median} sat std:{self.sat_std}')
        print(f'value mean:{self.value_mean}, value median:{self.value_median} value std:{self.value_std}')
   
    #low , hight
    # def _get_hue_scope(self)->Tuple[int,int]:
    #     low = self.hue_mean - 3 * self.hue_std
    #     low = max(30,low) # prevent low < 0
    #     high = self.hue_mean + 3 * self.hue_std
    #     high = min(80,high)  # prevent hight > 179
        
    #     if low >= high:
    #         high = low + 50

    #     return low,high

    # def _get_saturation_scope(self)->Tuple[int,int]:
    #     low = self.sat_mean - 2 * self.sat_std
    #     low = max(50, low)
    #     high = self.sat_mean + 4 * self.sat_std
    #     high = min(255,high)

    #     if low >= high:
    #         high = 255

    #     return low,high
    
    # def _get_value_scope(self)->Tuple[int,int]:
    #     # low = self.value_mean - 2 * self.value_std
    #     low = self.value_median - 3 * self.value_std
    #     low = max(50, low)
    #     # high = self.value_mean + 2 * self.value_std
    #     high = self.value_median + 4 * self.value_std
    #     high = min(255,high)

    #     if low >= high:
    #         high = 255

    #     return low,high
    
    # def get_hsv_scope(self)-> Tuple[List[int],List[int]]:
    #     if self.hue_mean == -1 :
    #         low_list = [30, 100, 100]
    #         high_list = [80, 255, 255]
    #     else:
    #         h_low,h_high = self._get_hue_scope()
    #         s_low,s_high =self._get_saturation_scope()
    #         v_low, v_high = self._get_value_scope()
    #         low_list = [h_low,s_low,v_low]
    #         high_list = [h_high,s_high,v_high]
    #     # for debug
    #     # print(f'*** low:{low_list} high:{high_list}')

    #     return low_list,high_list
    
    def is_valid(self)->bool:
        if self.hue_median < 30 or self.hue_median > 80:
            return False
        
        if self.sat_median < 50:
            return False 
        
        if self.value_median < 50:
            return False
        
        return True
#------------------------------------------------
class TennisBallInfoManager:
    def __init__(self):
        self.alph : float = 0.9
        self.hue_mean : int = 0
        self.hue_median : int = 0
        self.hue_std : int = 0

        self.sat_mean : int = 0
        self.sat_median : int = 0
        self.sat_std : int = 0

        self.value_mean :int = 0
        self.value_median :int = 0
        self.value_std : int = 0

        self.ball_info_list : List[TenisBallInfo] = []
        self.max_count = 10

    def show_info(self):
        print(f'h mean:{self.hue_mean}, hu median:{self.hue_median} h std:{self.hue_std}')
        print(f'sat mean:{self.sat_mean}, sat median:{self.sat_median} sat std:{self.sat_std}')
        print(f'value mean:{self.value_mean}, value median:{self.value_median} value std:{self.value_std}')
   
     #low , hight
     # 30<low < high < 80
    def _get_hue_scope(self)->Tuple[int,int]:
        # low = self.hue_median - 3 * self.hue_std
        low = self.hue_median - 25
        low = max(30,low) # prevent low < 0
        # high = self.hue_mean + 3 * self.hue_std
        high = self.hue_median + 25
        high = min(80,high)  # prevent hight > 179
        return low,high

    # 50 <= low < high <= 255
    # fast moving ball mix with white , so sat is low
    def _get_saturation_scope(self)->Tuple[int,int]:
        # low = self.sat_mean - 2 * self.sat_std
        low = self.sat_mean - 70
        low = max(50, low)
        # high = self.sat_mean + 4 * self.sat_std
        high = self.sat_mean + 70
        high = min(255,high)
        return low,high
    
    # 50 <= low < high <= 255
    def _get_value_scope(self)->Tuple[int,int]:
        # low = self.value_mean - 2 * self.value_std
        # low = self.value_median - 3 * self.value_std
        low = self.value_median - 70
        # low = self.value_median - 100
        # low = max(30, low)
        low = max(50, low)
        # high = self.value_mean + 2 * self.value_std
        # high = self.value_median + 4 * self.value_std
        high = self.value_median + 70
        high = min(255,high)

        return low,high
    
    def add_ball_info(self, ball_info:TenisBallInfo):
         #for debug
        # ball_info.show_info()

        if not ball_info.is_valid():
            return
        
        if len(self.ball_info_list) == 0 :
            self.hue_mean  = ball_info.hue_mean
            self.hue_median  = ball_info.hue_median
            self.hue_std  = ball_info.hue_std

            self.sat_mean = ball_info.sat_mean
            self.sat_median  = ball_info.sat_median
            self.sat_std  = ball_info.sat_std

            self.value_mean  = ball_info.value_mean
            self.value_median  = ball_info.value_median
            self.value_std  = ball_info.value_std
        else :
            self.hue_mean  = round(self.hue_mean * self.alph + (1-self.alph) * ball_info.hue_mean)
            self.hue_median  = round( self.hue_median * self.alph + (1-self.alph) * ball_info.hue_median)
            self.hue_std  = round(self.hue_std * self.alph + (1-self.alph) * ball_info.hue_std)

            self.sat_mean = round(self.sat_mean * self.alph + (1-self.alph) *ball_info.sat_mean)
            self.sat_median  = round( self.sat_median * self.alph + (1-self.alph) *ball_info.sat_median)
            self.sat_std  = round( self.sat_std * self.alph + (1-self.alph) *ball_info.sat_std)

            self.value_mean  = round(self.value_mean * self.alph + (1-self.alph) *ball_info.value_mean)
            self.value_median  = round(self.value_mean * self.alph + (1-self.alph) *ball_info.value_median)
            self.value_std  =  round( self.value_std * self.alph + (1-self.alph) *ball_info.value_std)

        self.ball_info_list.append(ball_info)

        if len(self.ball_info_list) > self.max_count:
            self.ball_info_list.pop(0)
        
     

    def get_hsv_scope(self)-> Tuple[List[int],List[int]]:
        if len(self.ball_info_list) == 0 :
            low_list = [30, 100, 100]
            high_list = [80, 255, 255]
        else:
            h_low,h_high = self._get_hue_scope()
            s_low,s_high =self._get_saturation_scope()
            v_low, v_high = self._get_value_scope()
            low_list = [h_low,s_low,v_low]
            high_list = [h_high,s_high,v_high]
            
        # for debug
        # print(f'*** low:{low_list} high:{high_list}')

        return low_list,high_list
    
    

      
# -----------------------------------------------
# record tennis ball info
class TenisBall:
    # two tennis ball at leas moving distance, continous
    MinMovingDistance = 5
    def __init__(self, center_x: int, center_y: int, width: int, height: int, area: int, step_count:int):
        # center_x, center_y is relative to target_roi,
        self.center_x = center_x
        self.center_y = center_y
        self.width = width
        self.height = height
        self.area = area
        self.step_count = step_count
        self.v_x = 0.0 
        self.v_y = 0.0 
        self.v_dot = 0.0

    def make_ball(self, center_point:Tuple[int,int]):
        x,y = center_point
        return TenisBall(center_x=x, center_y=y, width=self.width,height=self.height,area=self.area,step_count=self.step_count)
        
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

    def calculate_v(self, prev_ball: 'TenisBall')->bool:
        """
        if ball is close to previous, not move , static , return false,
        if move, in positin can distingush two ball return true
        """
        self.v_x = (self.center_x - prev_ball.center_x) 
        self.v_y = (self.center_y - prev_ball.center_y) 

        # normalize
        mag = math.sqrt(self.v_x * self.v_x + self.v_y * self.v_y)
        if mag < TenisBall.MinMovingDistance:
            return  False
        
        self.v_x = self.v_x / mag
        self.v_y = self.v_y / mag 
        # self.a_x = self.v_x - prev_ball.v_x
        # self.a_y = self.v_y - prev_ball.v_y
        # not usefull in jude direction change
        #self.v_cross =  self.v_y * prev_ball.v_x - self.v_x * prev_ball.v_y
        self.v_dot = self.v_x * prev_ball.v_x + self.v_y * prev_ball.v_y
        return True

    def show_v(self):
        return f"TenisBall(centerx={self.center_x}, centery={self.center_y}, v.x = {self.v_x}, v.y = {self.v_y}"

    def show_v_dot(self):
        return f"TenisBall(centerx={self.center_x}, centery={self.center_y}, v.x = {self.v_x}, v.y = {self.v_y}, v_dot={self.v_dot}"


# ---------------------------------------------------------------

class TennisBallManager:
    minCount = 6
    maxCount = 400
    maxStepCount = 60
    maxBallLastStepCount = 5
    minDirectionChanged = 0.7  # velocity dot product 

    def __init__(self):
        self.tennis_ball_path : List[TenisBall] = []
        self.last_ball_step_count = 0   # ball in target roi  no more than step count 20
        self.is_hit = False  # if hit , do not do hit test for following find ball
        self.hit_step_count = 0 # hit time

        # self.tennis_ball_info  = TenisBallInfo()
        self.is_need_tennis_ball_info_update = True
        self.tennis_ball_info_manager  = TennisBallInfoManager()

        # 统计检测到球的次数
        self.count_ball_detect = 0
        self.count_ball_detect_in_black_score_10 = 0
    
    def _should_reset_ball_path(self,step_count):
        if step_count - self.last_ball_step_count > TennisBallManager.maxStepCount:
            # print("******* clear  tennis ball path *********")
            self.tennis_ball_path.clear()

    def  _should_reset_hit(self, step_count):
        if step_count - self.hit_step_count > TennisBallManager.maxStepCount:
            self.is_hit = False
            self.hit_step_count = step_count


    # 设置网球信息，从黑10目标检测出来
    def set_tennis_ball_info(self, ball_info: TenisBallInfo):
        self.count_ball_detect_in_black_score_10 += 1
        print(f"count roi :{self.count_ball_detect}  count roi black score: {self.count_ball_detect_in_black_score_10} ")
        
        self.is_need_tennis_ball_info_update = False
        # self.tennis_ball_info = ball_info

    def add_tennis_ball_info(self, ball_info: TenisBallInfo):
        self.is_need_tennis_ball_info_update = False
        self.tennis_ball_info_manager.add_ball_info(ball_info=ball_info)

    # 根据上帧网球得到当前需检测的中心位置
    def get_center_for_current_target_roi(self, step_count :int )->Tuple[int,int] | None:
        if step_count - self.last_ball_step_count > 3:
            return None # lost ball, should search all target roi
        
        if len(self.tennis_ball_path) == 0:
            return None
        
        last_ball = self.tennis_ball_path[-1]
        return last_ball.get_center()


    # ball hsv low upper range 
    def _get_tennis_ball_range(self)->tuple[np.ndarray,np.ndarray]:
            # lower_list ,high_list = self.tennis_ball_info.get_hsv_scope()
            lower_list ,high_list = self.tennis_ball_info_manager.get_hsv_scope()
            lower_green = np.array(lower_list)
            upper_green = np.array(high_list)

            return lower_green,upper_green


     # according info, give a score , ball like 
    def _get_ball_like_score(self, left, top, width,height, area, centroid_x, centroid_y)->int :
        score = 0 

        if  10 <= area < 100:
            score += 10

        if  100 <= area  < 500:
            score += 20

        if width * height < area * 2:
            score += 20
        
        # print(f'jude ball score:{score}')
        return score
    
    # if more than 2 ball like blob, only check  is in target circle
    def _get_ball_like_in_target_circle_score(self, left, top, width,height, area, centroid_x, centroid_y,target_manager)->int :
      
        if  target_manager.check_ball_like_center_in_target_circle((centroid_x,centroid_y)):
            return self._get_ball_like_score( left, top, width,height, area, centroid_x, centroid_y)
        else:
            return 0
        
    
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
    
    # find ball in max target roi 
    #deprecated, not used 
    def find_ball(self, roi:np.ndarray, step_count:int)->bool:

      
        self._should_reset_hit(step_count=step_count)


        green_binary = self.get_green_from_roi(roi)
        count = np.count_nonzero(green_binary)
        # print(f'ball area count:{count}')
        # if count < TennisBallManager.minCount or count > TennisBallManager.maxCount:
        #     return False
        if count < TennisBallManager.minCount :
            return False
        
        # found ball 
        # print(f" find ball : {count}")
        best_left ,best_top = 0, 0
        best_width, best_height = 0,0
        best_area , best_label = 0, 0
        best_centroid_x , best_centroid_y = 0,0 
        best_score = 0

        # 进行连通组件分析
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(green_binary, connectivity=8)
        # 如果出现多个识别块，识别 信息有问题，球撞击幕布，幕布变形
        if num_labels > 4 :
            # self.tennis_ball_info = TenisBallInfo()
            return False
               
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
        if best_score == 0 : #没有识别到球
            return False
        
        # the tennis ball appear in target region time 
        # 网球间隔 2.5秒 ，超过maxStepcount 认为是新球
        self._should_reset_ball_path(step_count=step_count)

        # 识别到球
        x = int(best_centroid_x)
        y = int(best_centroid_y)
        w = int(best_width)
        h = int(best_height)
        a = int(best_area)
        ball = TenisBall(center_x=x, center_y=y,width= w,height=h,area= a,step_count=step_count)
        self.last_ball_step_count = step_count
        self.count_ball_detect += 1

        if len(self.tennis_ball_path) > 0:
            prev_ball = self.tennis_ball_path[-1]
            #if ball moving too small, may static, can not caculate velocity
            if ball.calculate_v(prev_ball=prev_ball):
                self.tennis_ball_path.append(ball)  #can distinguish with previous ball in position
        else :
            self.tennis_ball_path.append(ball)  #first found ball 

        # print(ball)
        # print(ball.show_v_dot())
     
        return True
    
    def find_ball_in_current_target_roi(self, current_target_roi:np.ndarray,  step_count:int, target_manager)->TenisBall|None:

        self._should_reset_hit(step_count=step_count)

        green_binary = self.get_green_from_roi(current_target_roi)

        count = np.count_nonzero(green_binary)
        # for debug
        # print(f'ball area count:{count}')
        # if count < TennisBallManager.minCount or count > TennisBallManager.maxCount:
        #     return False
        if count < TennisBallManager.minCount :
            return None
        

        # found ball 
        # print(f" find ball : {count}")
        best_left ,best_top = 0, 0
        best_width, best_height = 0,0
        best_area , best_label = 0, 0
        best_centroid_x , best_centroid_y = 0,0 
        best_score = 0

        # 进行连通组件分析
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(green_binary, connectivity=8)
        # 如果出现多个识别块，识别 信息有问题，球撞击幕布，幕布变形
        # 0-background 1,2,3 ... --foreground
        # print(f'num_labels:{num_labels} ')
        # if num_labels > 4 :
        if num_labels > 6 :
            # self.tennis_ball_info = TenisBallInfo()
            print(f"too many ball like object : {num_labels}")
            # return None
        #forground has more than 2 blob, only in target circle is valid, discard out of target ball like cluster

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

            if num_labels == 2:  # only one foreground ball like
                score = self._get_ball_like_score(left=left,top=top,width=width,height=height,area=area,centroid_x=centroid_x,centroid_y=centroid_y)
            else:  # 2 or more detected, only in target circle is valide
                score = self._get_ball_like_in_target_circle_score(left=left,top=top,width=width,height=height,area=area,centroid_x=centroid_x,centroid_y=centroid_y,target_manager=target_manager)

            if score > best_score:
                best_score = score
                best_left ,best_top = left, top
                best_width, best_height = width,height
                best_area , best_label = area, label
                best_centroid_x , best_centroid_y = centroid_x,centroid_y

        # found tennis ball 
        if best_score == 0 : #没有识别到球
            return None
        
        # the tennis ball appear in target region time 
        # 网球间隔 2.5秒 ，超过maxStepcount 认为是新球
        self._should_reset_ball_path(step_count=step_count)
        
        # 识别到球
        x = int(best_centroid_x)
        y = int(best_centroid_y)
        w = int(best_width)
        h = int(best_height)
        a = int(best_area)

        ball = TenisBall(center_x=x, center_y=y,width= w,height=h,area= a,step_count=step_count)
        self.last_ball_step_count = step_count
        self.count_ball_detect += 1
        mask = labels == best_label
        ball_info = self._compute_ball_info(origin_image=current_target_roi,mask=mask)
        self.tennis_ball_info_manager.add_ball_info(ball_info=ball_info)
        # for debug
        # ball_info.show_info() 
        
        return ball

    
    # ball is find a black score 10 by know little info
    # ball is find in current target roi
    # output : True add success False no add, because ball is static 
    def add_ball(self, ball:TenisBall)->bool:
        if len(self.tennis_ball_path) > 0:
            prev_ball = self.tennis_ball_path[-1]
            #if ball moving too small, may static, can not caculate velocity
            if ball.calculate_v(prev_ball=prev_ball):
                self.tennis_ball_path.append(ball)  #can distinguish with previous ball in position
                return True
            else:
                return False
        else :
            self.tennis_ball_path.append(ball)  #first found ball 
            return True

    def get_last_ball(self)->Tuple[int,int,int,int]:
        ball = self.tennis_ball_path[-1]
        return (ball.center_x, ball.center_y, ball.width, ball.height)
    
    #hit convas test
    def hit_test(self, step_count, is_hit_canvas:HitCanvas)->Tuple[int,int,bool] | None:
        if self.is_hit:
            # print(f"is_hit: step_count:{step_count}  is_hit_canvas:{is_hit_canvas}")
            return None
        
        if len(self.tennis_ball_path) == 0 :
            # print(f"no ball : is_hit:{self.is_hit}, step_count:{step_count}  is_hit_canvas:{is_hit_canvas}")
            if is_hit_canvas == HitCanvas.HitCanvas:
                self.is_hit = True
                self.hit_step_count = step_count

            return None
        
        # print(f" ball : {len(self.tennis_ball_path)} is_hit:{self.is_hit}, step_count:{step_count}  is_hit_canvas:{is_hit_canvas}")
        
        # exceed  maxBallLastStepCount frame no ball found, return last ball as hit 
        # if (step_count - self.last_ball_step_count) > TennisBallManager.maxBallLastStepCount:
        #     last_ball = self.tennis_ball_path[-1]
        #     x = last_ball.center_x
        #     y = last_ball.center_y
        #     self.is_hit = True
        #     # print("hit  exceed  maxBallLastStepCount")
        #     return (x,y, True)

        # detect hit canvas by hit manager throught background substract
        # I am sure : ball is hit canvas, because threshold is set too hight,
        # slow mode
        if is_hit_canvas == HitCanvas.HitCanvas:
            last_ball = self.tennis_ball_path[-1]
            x = last_ball.center_x
            y = last_ball.center_y
            self.is_hit = True
            self.tennis_ball_path.clear()
            # self.hit_step_count = last_ball.step_count
            self.hit_step_count = last_ball.step_count
            # print(f"hit_canvas : x={x } y={y} input step_count:{step_count}  hit_step_count :{self.hit_step_count}")
            return (x,y, True)
        
        if is_hit_canvas == HitCanvas.NotHitCanvas:
            # print("not hit canvas ")
            return None
        
        # print(f"not sure hit canvas balls:{len(self.tennis_ball_path)}")

        # I am not sure, ball hit canvas, need another way to detect
         # exceed  maxBallLastStepCount frame no ball found, return middle ball as hit 
        if (step_count - self.last_ball_step_count) > TennisBallManager.maxBallLastStepCount:
            # mid_index = len(self.tennis_ball_path) // 2
            # mid_ball = self.tennis_ball_path[mid_index]
            # x = mid_ball.center_x
            # y = mid_ball.center_y
            # self.is_hit = True
            # self.hit_step_count =  mid_ball.step_count
            # # print("hit  exceed  maxBallLastStepCount")
            # return (x,y, True)

            #case 1 : only 1 or 2 ball 
            if len(self.tennis_ball_path) < 3:
                last_ball = self.tennis_ball_path[-1]
                x = last_ball.center_x
                y = last_ball.center_y
                self.is_hit = True
                self.hit_step_count = last_ball.step_count
                return (x,y, True)
            
            #case 2 : more than 2 ball, 
            min_v_dot = 2
            min_ball = self.tennis_ball_path[2]

            for ball in self.tennis_ball_path[2:]:
                if ball.v_dot < min_v_dot:
                    min_v_dot = ball.v_dot
                    min_ball = ball

            # ball hit outside canvas, fall pass through target circle
            # if no direction change, do not score
            # x = min_ball.center_x
            # y = min_ball.center_y
            # self.is_hit = True
            # self.hit_step_count = min_ball.step_count
            # return (x,y, True)
            if min_v_dot < TennisBallManager.minDirectionChanged:
                x = min_ball.center_x
                y = min_ball.center_y
                self.is_hit = True
                self.hit_step_count = min_ball.step_count
                return (x,y, False)
            else:
                return None
            
        # only one ball detected
        if len(self.tennis_ball_path) < 2:
            return None  # less to detect hit 
        
        # found ball - lost ball - lost ball - bost ball -- found ball
        # fly over white  interference
        last_ball_1 = self.tennis_ball_path[-1]
        last_ball_2 = self.tennis_ball_path[-2]

        # has lost some frame between continue ball 
        if (last_ball_1.step_count - last_ball_2.step_count) > 2:
            x = last_ball_2.center_x
            y = last_ball_2.center_y
            self.is_hit = True
            self.hit_step_count = last_ball_2.step_count
            # print("hit white, discontinure frame found ball , fly in white and fly out white")
            return (x,y,True)
        
        # only 3 balls position ->  two velocity -> direction change 
        if len(self.tennis_ball_path) > 2:
            if last_ball_1.v_dot < TennisBallManager.minDirectionChanged:
                # direction change 
                x = last_ball_2.center_x
                y = last_ball_2.center_y
                self.is_hit = True
                self.hit_step_count = last_ball_2.step_count
                # print("hit direction change")
                return (x,y,False)
            
        return None
        
    # key to find ball 
    def get_green_from_roi(self, roi: np.ndarray) -> np.ndarray:

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        lower_green, upper_green = self._get_tennis_ball_range()
        mask = cv2.inRange(hsv, lower_green, upper_green)

        filtered_image = cv2.medianBlur(mask, 3)

        # kernel = np.ones((3,3), np.uint8)
        kernel = np.ones((5,5), np.uint8)
        opened_bitmap = cv2.morphologyEx(filtered_image, cv2.MORPH_OPEN, kernel)
        return  opened_bitmap
    
   

    # in hsv , get green look like  area
    def get_green_from_roi_background(self, roi: np.ndarray) -> np.ndarray:

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        lower_white = np.array([0, 0, 100])
        upper_white = np.array([179, 50, 255])

        # 使用 cv2.inRange 函数创建一个二值图像，其中绿色像素为白色，其他为黑色
        mask = cv2.inRange(hsv, lower_white, upper_white)
        return mask
    
        filtered_image = cv2.medianBlur(mask, 3)

        kernel = np.ones((3,3), np.uint8)
        opened_bitmap = cv2.morphologyEx(filtered_image, cv2.MORPH_OPEN, kernel)
        return  opened_bitmap
    
     
  


   
 

  
