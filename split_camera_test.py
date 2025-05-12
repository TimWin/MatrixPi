import cv2
import Camera
import time
import math
import numpy as np
from devices.hailo import Hailo
import threading

# Camera test
camera = Camera.Camera(resolution=(1280, 720))
camera.camera_open(correction=False)
img = camera.frame
#planes
left_plane = (0, int(camera.height), 0, int(camera.width/2))
right_plane = (0, camera.height, int(camera.width/2), camera.width)

#capture = cv2.VideoCapture(-1)
#capture.set(cv2.CAP_PROP_FPS, 30)


#HAILO
#hailo = Hailo("models/compiled/hand_landmark.hef")
#model_h, model_w, _ = hailo.get_input_shape()
#print(model_h, " x ", model_w)

#ui
finish_process = False
def show_frame():
	global img
	global finish_process
	while True:
		if img is not None:
			cv2.imshow('frame', img)
			key = cv2.waitKey(1)
			if key == 27:
				finish_process = True
				break
			time.sleep(0.03333)
	cv2.destroyAllWindows()

th = threading.Thread(target=show_frame)
th.setDaemon(True)
th.start()

#main
prev_time = time.perf_counter()
while not finish_process:
	current_time = time.perf_counter()
	elapsed_time = (current_time - prev_time) * 1000.0
	prev_time = current_time
	print(f"Frame processing took {elapsed_time:.6f} ms")
	
	img = camera.frame
	if img is not None:
		frame = img.copy()
		left_image = frame[right_plane[0]:right_plane[1], right_plane[2]:right_plane[3]]
		#cv2.imshow('frame', left_image)
		#time.sleep(0.1)
		
		
#hailo.close()
camera.camera_close()
