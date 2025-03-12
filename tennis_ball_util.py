import numpy as np
import cv2
from typing import Tuple, Optional, List
import math

# I study the my tennis ball detect method, it is hard to further optimize.
# my method can be break inot 3 steps:
# 1. frmae BGR -> HSV,  opencv buildin method is optimize and fast 
# 2 . hsv to binary image, and open operation  morphologyEx, this operation is not local, so need scan around
# 3.  binary image to find ball, need points aggreation, 
# it is hard to make 3 steps into one step , so I give up to optimize for time being.

# will come from config.json
# target canvas scope
min_x = 80
min_y = 63 
max_x = 601
max_y = 403

# tenis ball - area - points count
min_area = 20 
max_area = 500

# background bright : remove background black
# range: 0 to 255
# min_value = 60 
# max_value = 120
min_value = 80
max_value = 200

# range: 0 to 179, green 60
# tennis ball color: can distinguish from canvas background
min_hue = 30
max_hue = 50
# min_hue = 20
# max_hue = 60
# min_hue = 50
# max_hue = 70
# remove background white
max_saturation = 40 

def make_binary_bitmap_from_frame(frame: np.ndarray) -> np.ndarray:

    # blur_frame = cv2.GaussianBlur(frame,(11, 11), 0)
    # blur_frame = cv2.medianBlur(frame,5)
      # Convert the frame to HSV format
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Split the HSV frame into its channels
    h, s, v = cv2.split(hsv_frame)

    # Apply a threshold to the value channel to create a binary bitmap
    # _, binary_bitmap = cv2.threshold(v, 128, 1, cv2.THRESH_BINARY)
    return  make_binary_bitmap(h,s,v)


def make_binary_bitmap(h: np.ndarray, s: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Create a binary bitmap based on hue and value thresholds.

    Parameters:
    h (np.ndarray): Hue channel of the image.
    s (np.ndarray) : staturation channel of image
    v (np.ndarray): Value channel of the image.
   

    Returns:
    np.ndarray: Binary bitmap.
    """

    # please aution to use ai to optimize, I try several time but fialure
    binary_bitmap = np.ones_like(v)

    for (i, j), value in np.ndenumerate(v):
        if i < min_y or i > max_y :
            binary_bitmap[i,j] = 0
            continue 

        if j < min_x or j > max_x :
            binary_bitmap[i,j] = 0
            continue

        h_value = h[i, j]
        s_value = s[i,j]

        # white
        if s_value < max_saturation :
            binary_bitmap[i,j] = 0
            continue

        if h_value > max_hue or h_value < min_hue:
            binary_bitmap[i, j] = 0
        elif value > max_value or value < min_value:
            binary_bitmap[i, j] = 0

         # Set binary_bitmap to 1 where the conditions are met
        #  binary_bitmap[(v >= min_value) & (v <= max_value) & (h >= min_hue) & (h <= max_hue)] = 1
    # open : erode and dilate, remove noise 
    # kernel = np.ones((5,5), np.uint8)
    kernel = np.ones((3,3), np.uint8)
    opened_bitmap = cv2.morphologyEx(binary_bitmap, cv2.MORPH_OPEN, kernel)


    return opened_bitmap


# record tennis ball info
class TenisBall:
    def __init__(self, centerx: int, centery: int, width: int, height: int, area: int, step_count:int):
        self.centerx = centerx
        self.centery = centery
        self.width = width
        self.height = height
        self.area = area
        self.step_count = step_count
        self.v_x = 0.0 
        self.v_y = 0.0 
        self.a_x = 0.0
        self.a_y = 0.0
        self.v_cross = 0.0
        self.v_dot = 0.0

    def __repr__(self):
        return f"TenisBall(centerx={self.centerx}, centery={self.centery}, width={self.width}, height={self.height}, area={self.area} ,step_count={self.step_count})"

    def get_center(self):
        return (self.centerx, self.centery)

    def get_dimensions(self):
        return (self.width, self.height)

    def get_area(self):
        return self.area
    
    def get_setup_count(self):
        return self.step_count

    def calculate_v_a(self, prev_ball: 'TenisBall'):
        step = self.step_count - prev_ball.step_count
        if step == 0:
            return
        self.v_x = (self.centerx - prev_ball.centerx) / step
        self.v_y = (self.centery - prev_ball.centery) / step
        # normalize
        mag = math.sqrt(self.v_x * self.v_x + self.v_y * self.v_y)
        self.v_x = self.v_x / mag
        self.v_y = self.v_y / mag 
        # self.a_x = self.v_x - prev_ball.v_x
        # self.a_y = self.v_y - prev_ball.v_y
        self.v_cross =  self.v_y * prev_ball.v_x - self.v_x * prev_ball.v_y
        self.v_dot = self.v_x * prev_ball.v_x + self.v_y * prev_ball.v_y

    def show_v_a(self):
        a = math.sqrt( self.a_x * self.a_x + self.a_y * self.a_y)
        return f"TenisBall(centerx={self.centerx}, centery={self.centery}, v.x = {self.v_x}, v.y = {self.v_y},a.x = {self.a_x} ,a.y= {self.a_y} ,a= {a}"

    def show_v_cross(self):
        return f"TenisBall(centerx={self.centerx}, centery={self.centery}, v.x = {self.v_x}, v.y = {self.v_y}, v_cross = {self.v_cross}, v_dot={self.v_dot}"


def add_point(list:List[Tuple[int,int]], point:Tuple[int,int])-> bool:
    if len(list) == 0 :
        list.append(point)
        return True
    
    for p in list:
        if abs(point[0] - p[0]) + abs(point[1] - p[1]) < 4:
            # neighbour point 
            list.append(point)
            return True
        
    return False
    

def find_tenis_ball(binary_image: np.ndarray,step_count:int) -> Tuple[bool, Optional[TenisBall]]:
    """
    Find the tennis ball in a binary image.

    Parameters:
    binary_image (np.ndarray): Binary image where the tennis ball is to be found.

    Returns:
    Tuple[bool, Optional[TenisBall]]: A tuple containing a boolean indicating if a tennis ball was found,
    and an instance of TenisBall if found, otherwise None.
    """
    # aggreation points
    points_list : List[List[Tuple[int,int]]] = []

    # all points that value = 1
    points =  np.argwhere(binary_image == 1)
    area = len(points)

    # not found ball 
    if area < min_area :
         return False, None
    
    # classify the indices
    for p in points:
        is_add = False
        for list in points_list:
            ret = add_point(list,p)
            if ret:
                is_add = True
                break

        if not is_add:
           l = [p]
           points_list.append(l) 

    for lst in points_list:
        area = len(lst)
        if area > min_area and area < max_area:
            # Split the list of tuples into two lists
            y_coords, x_coords = zip(*lst)

            # Calculate the center and dimensions of the detected object
            center_y = np.mean(y_coords)
            center_x = np.mean(x_coords)
            min_y = np.min(y_coords)
            max_y = np.max(y_coords)
            height = max_y - min_y

            min_x = np.min(x_coords)
            max_x = np.max(x_coords)
            width = max_x - min_x

            # Create a TenisBall instance
            tennis_ball = TenisBall(centerx=int(center_x), centery=int(center_y), width=int(width), height=int(height), area=area, step_count=step_count)
            return True, tennis_ball

    else:
        return False, None
# if v_x, v_y is normlized, it do not work 

def tennis_ball_hit_test_deprecated(lst:List[TenisBall])->bool:
    """
    list  of tenis ball detect
    return : hit : true, false : no hit
    """
    # too short to test
    if len(lst) < 4 :
        return False
    
    # too long , ball hit target on last short time
    if len(lst) > 40 :
        return False
    
    last_2 = lst[-2]
    last_1 = lst[-1]

    change = last_1.v_x * last_2.v_x 

    return change < -1

    
def tennis_ball_hit_test(lst:List[TenisBall])->bool:
    """
    list  of tenis ball detect
    return : hit : true, false : no hit
    """
    # too short to test
    if len(lst) < 4 :
        return False
    
    # too long , ball hit target on last short time
    if len(lst) > 40 :
        return False
    
    last_2 = lst[-2]
    last_1 = lst[-1]

    change1 =  math.fabs( math.fabs(last_2.v_cross) - math.fabs(last_1.v_cross) )
    change12 = math.fabs( last_2.v_dot - last_1.v_dot )
    change = change1 + change12
    print(f'change: {change}')

    return change > 0.5
    
    
