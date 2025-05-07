import cv2
import numpy as np
from enum import Enum

class HitCanvas(Enum):
    HitCanvas = 2
    NotSureHitCanvas = 1
    NotHitCanvas = 0



class HitManager():
    Max_Fgbg_Count = 100
    Min_Fgbg_Count = 10
    Max_Fgbg_Len = 100

    def __init__(self):
        # 创建背景减除器对象
        self.fgbg = cv2.createBackgroundSubtractorMOG2(history=60, varThreshold=100, detectShadows=False)
        # self.fgbg = cv2.createBackgroundSubtractorMOG2(history=60, varThreshold=200, detectShadows=False)
        # fgbg = cv2.createBackgroundSubtractorKNN()
        # mog2 = cv2.createBackgroundSubtractorMOG2(history=50, varThreshold=20, detectShadows=False)
        # knn = cv2.createBackgroundSubtractorKNN(history=200, dist2Threshold=200.0, detectShadows=False)
        #dynamic threshold
        self.fgbg_mask_list = []
        self.count = 0 
        # self.mean = 0
        # self.std = 0 
        self.max_fgbg_count = HitManager.Max_Fgbg_Count
        self.min_fgbg_count = HitManager.Min_Fgbg_Count
        self.outlier_count = HitManager.Max_Fgbg_Count * 2

    # deprecated
    def apply(self, target_roi, frame_step):
        resized_image = cv2.resize(target_roi, None, fx=0.2, fy=0.2, interpolation=cv2.INTER_AREA)
        # 应用背景减除器
        fgmask = self.fgbg.apply(resized_image)
        sum = np.count_nonzero(fgmask)
        # self.fgbg_count_list.append(sum)
        # index = len(self.fgbg_count_list) - 1
        # self.frame_step_dict[frame_step] = index
        # print(f"hit sum: {sum}")
    
    def reset(self):
        pass
        # self.fgbg_count_list.clear()
        # self.frame_step_dict.clear()

    def _should_update_threshold(self):
        if self.count < HitManager.Max_Fgbg_Len:
            return
        
        arr = np.array(self.fgbg_mask_list)
        _mean = arr.mean()
        _std = arr.std()
        self.count = 0 
        self.fgbg_mask_list.clear()
        # print(f"mean:{_mean} std:{_std}")

        if _std > 10 :
            self.max_fgbg_count = _mean + 2 * _std
            self.outlier_count = _mean + 4 * _std
            self.min_fgbg_count = max  ( HitManager.Min_Fgbg_Count, _mean  )



    def test_hit(self, target_roi)->HitCanvas:
        resized_image = cv2.resize(target_roi, None, fx=0.2, fy=0.2, interpolation=cv2.INTER_AREA)
        # 应用背景减除器
        fgmask = self.fgbg.apply(resized_image)
        sum = np.count_nonzero(fgmask)
        # print(f"fgbg count:{sum}")
        if 1 < sum < self.outlier_count :  # less 1 cansider noise ,outlier not include
            self.fgbg_mask_list.append(sum)
            self.count += 1
            self._should_update_threshold()

        if sum > self.outlier_count:
            return HitCanvas.NotHitCanvas
        elif  sum > self.max_fgbg_count:
            return HitCanvas.HitCanvas
        elif sum < self.min_fgbg_count:
            return HitCanvas.NotHitCanvas
        else:
            return HitCanvas.NotSureHitCanvas
            

        


