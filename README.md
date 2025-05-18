# 基于yolo11的网球落点检测项目（部署于香橙派5max）

![tennis_logo](https://github.com/user-attachments/assets/1f677a89-d2d2-4111-8825-0d71ab92f9a4)

> 请勿 fork 或者 clone 本仓库，仅个人使用

## Update 1.13
甲方验货前已完成板端rknn的部署，但是运行帧率略低，下面的优化有待实现：

- 训练.pt模型时，将激活函数改为ReLU✅
- 板端推理时使用线程池提高NPU利用率❓

目前有三点问题：

- 帧率未达到60帧
- 识别靶标的参数没调好（能不能自动调）
- 识别撞击不精确

## Update 1.22

针对之前的三点问题，作了如下优化：

- 使用 2 个进程解耦了摄像头进程和 Yolo 检测进程，目前实时帧率可以达到 60 帧
- 简化了靶标识别逻辑，取消了颜色筛选，使用形状检测，解决了之前光照会影响检测的问题
- 使用网球检测框大小变化作为碰撞判据，当网球检测框大小处于全局最小值时判定为碰撞
- Yolo 进程内使用多线程并发获取 frame 并检测，同时使用 reorder_buffer 确保检测结果按顺序提交

目前存在的问题：

- 网球检测框显示在画面上存在延迟（因为使用了多进程）
- 碰撞识别还可以优化，提高准确率

## Update 1.30

针对之前的问题，作了如下修改：

- 使用**多进程**检测取代**多线程**进行yolo检测，实测是降低了检测框追踪延迟 
- 在碰撞判断中加入了某极值是否是全局前三小的判断，防止某些噪音影响判断

感觉已经接近最终版本了，看甲方那边什么情况

###############################################################3
#新版1.2 不使用yolo 深度学习网络来识别网球。
原因:
1. 网球特征太少，用深度学习没有意义
2. yolo学习没有使用场景的图片，并做了网球标记,没有针对使用场景进行学习。
3. 使用深度网络效率低。
4. 快速运动网球图像是模糊的， 容易与白色背景融合，不能识别出来。
对使用场景要求：
1.网球要绿   2.背景布尽量不要使用白色。

#新版1.4 
1.首先检测目标框
2.检测目标框黑10目标框，得到黑10的背景颜色信息(hue, saturation. value)
3.在黑10上做网球检测，通过背景删除，留下的为网球，通过联通性分析，得到可能 网球
4. 分析网球颜色信息， 为在目标框检测网球提供信息
5. 在目标框检测网球
6. 分析网络路径计算撞击点， 
7. 计算撞击点所在分值


项目文件列表:
README.md
constants.py
main.py
tools.py
network.py
target_manager.py
tennis_ball_manager.py
tennis_logo.png
requirements.txt
meta/config.json
meta/settings.json
hit_manager.py

11

第三方依赖包:
numpy
Requests==2.32.3
opencv-python==4.10.0.84

Press h --> relocateTarget --> find target -> targetroi
                                          -> black score 10 roi -> need update background


        --> need update ball info 

version 2.0 
hit_manager.py : use background substraction detect ball hit canvas
roi -> resize -> count non zero :  not hit, not sure, hit happen, outlier

frame rate > 100, detect many ball like object , noise,
make a reasonable frame rate is important, now frame rate at 40

version 2.3
when hit target is detect, do not detect ball for 1.5 seconds, because light up
this will effect  background subtraction statistics, so ignore it for 1.5 seconds

