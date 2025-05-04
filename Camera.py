#!/usr/bin/env python3
# encoding:utf-8
import sys
import cv2
import time
import threading
import numpy as np
from CameraCalibration.CalibrationConfig import *

# 调用USB摄像头(call USB camera)

if sys.version_info.major == 2:
    print('Please run this program with python3!')
    sys.exit(0)

class Camera:
    def __init__(self, resolution=(640, 480)):
        self.cap = None
        self.width = resolution[0]
        self.height = resolution[1]
        self.frame = None
        self.opened = False
        
        #加载参数(load parameter)
        self.param_data = np.load(calibration_param_path + '.npz')
        #获取参数(get parameter)
        dim = tuple(self.param_data['dim_array'])
        k = np.array(self.param_data['k_array'].tolist())
        d = np.array(self.param_data['d_array'].tolist())
        p = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(k, d, dim ,None).copy()
        self.map1, self.map2 = cv2.fisheye.initUndistortRectifyMap(k, d, np.eye(3), p, dim, cv2.CV_16SC2)
        
        self.th = threading.Thread(target=self.camera_task, args=(), daemon=True)
        self.th.start()

    def camera_open(self, correction=False):
        try:
            self.cap = cv2.VideoCapture(-1)
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('Y', 'U', 'Y', 'V'))
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.cap.set(cv2.CAP_PROP_SATURATION, 40)
            self.correction = correction
            self.opened = True
        except Exception as e:
            print('打开摄像头失败:', e)

    def camera_close(self):
        try:
            self.opened = False
            time.sleep(0.2)
            if self.cap is not None:
                self.cap.release()
                time.sleep(0.05)
            self.cap = None
        except Exception as e:
            print('关闭摄像头失败:', e)

    def camera_task(self):
        while True:
            try:
                if self.opened and self.cap.isOpened():
                    ret, frame_tmp = self.cap.read()
                    if ret:
                        frame_resize = cv2.resize(frame_tmp, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
                        
                        if self.correction:
                            self.frame = cv2.remap(frame_resize, self.map1, self.map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                        else:
                            self.frame = frame_resize
                            
                        ret = False
                    else:
                        self.frame = None
                        self.cap.release()
                        cap = cv2.VideoCapture(-1)
                        ret, _ = cap.read()
                        if ret:
                            self.cap = cap
                elif self.opened:
                    self.cap.release()
                    cap = cv2.VideoCapture(-1)
                    ret, _ = cap.read()
                    if ret:
                        self.cap = cap              
                else:
                    time.sleep(0.01)
            except Exception as e:
                print('获取摄像头画面出错:', e)
                time.sleep(0.01)

if __name__ == '__main__':
    camera = Camera()
    camera.camera_open()
    while True:
        img = camera.frame
        if img is not None:
            cv2.imshow('img', img)
            key = cv2.waitKey(1)
            if key == 27:
                break
    my_camera.camera_close()
    cv2.destroyAllWindows()
