#!/usr/bin/env python3
# encoding:utf-8
# Data:2021/05/25
# Author:Aiden
# Function: 采集标定图像(collect calibration images)
import os
import cv2
import time
from CalibrationConfig import *

print('按下键盘上的space键存储图像，按esc退出')
cap = cv2.VideoCapture(-1)

pictures_list = []
#如果calib文件夹不存在，则新建(create a new 'calib' folder if it does not exist)
if not os.path.exists(save_path):
    os.mkdir(save_path)
else:
    for i in os.listdir(save_path):
        pictures_list.append(i[:-4])

#计算存储的图片数量(calculate the number of stored images)
num = 0
while True:
    ret, frame = cap.read()
    if ret:
        Frame = frame.copy()
        cv2.putText(Frame, str(num), (10, 50), cv2.FONT_HERSHEY_COMPLEX, 2.0, (0, 0, 255), 5)
        cv2.imshow("Frame", Frame)
        key = cv2.waitKey(1)
        if key == 27:
            break
        if key == 32:
            while True:
                num += 1
                if num not in pictures_list:
                    #图片名称格式：当前图片数量.jpg(image name format: current image count.jpg)
                    cv2.imwrite(save_path + str(num) + ".jpg", frame)
                    break
    else:
        time.sleep(0.01)

cap.release()
cv2.destroyAllWindows()
