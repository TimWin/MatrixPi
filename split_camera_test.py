import cv2
import Camera
import time
import math
import numpy as np
from devices.hailo import Hailo
import threading
import GestureDetection as gd

from Video.FrameStream import FrameStreamer

# Camera test
camera = Camera.Camera(resolution=(1280, 720))
camera.camera_open(correction=False)
img = camera.frame

#HAILO
hailo = Hailo("models/compiled/hand_landmark.hef")
model_h, model_w, _ = hailo.get_input_shape()
#print(model_h, " x ", model_w

#MODEL
hand_present = "hand_landmark/fc2"
handness = "hand_landmark/fc4"
normalized_locations = "hand_landmark/fc3"
world_locations = "hand_landmark/fc1"

#DETECTOR
detector_left = gd.GestureDetection(0.5)
detector_right = gd.GestureDetection(0.5)

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

#th = threading.Thread(target=show_frame)
#th.setDaemon(True)
#th.start()

def pre_process_img(img, width, height):
	return cv2.resize(img, (width, height)).copy()
	
def run_ai(img, detector):
	results = hailo.run(img)
	hand_presence = results[hand_present]
	hand_direction = results[handness]
	landmarks_world = results[world_locations]
	
	detector.update(hand_presence, hand_direction, landmarks_world)
	detector.draw_gesture(img, landmarks_world)

#main

#Stream
frame_streamer = FrameStreamer()
frame_streamer.start(host='0.0.0.0', port=5000)


prev_time = time.perf_counter()
while not finish_process:
	current_time = time.perf_counter()
	elapsed_time = (current_time - prev_time) * 1000.0
	prev_time = current_time
	#print(f"Frame processing took {elapsed_time:.6f} ms")
	
	raw_img = camera.get_next_frame_blocking() #frame
	if raw_img is not None:
		raw_img = cv2.flip(raw_img, 1)
		height, width, _ = raw_img.shape
		left_img = raw_img[:, :width // 2]
		right_img = raw_img[:, width // 2:]
		
		processed_left = pre_process_img(left_img, 224, 224)
		processed_right = pre_process_img(right_img, 224, 224)
		
		run_ai(processed_left, detector_left)
		run_ai(processed_right, detector_right)
		
		print("Gesture left: ", detector_left.detected_gesture, ", right: ", detector_right.detected_gesture)
		
		img = np.hstack((processed_left, processed_right))
		frame_streamer.update_frame(img)
		
		#cv2.imshow('frame', left_image)
		#time.sleep(0.1)
		
		
hailo.close()
camera.camera_close()
streamer_thread.stop()
