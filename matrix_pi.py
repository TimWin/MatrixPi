import Camera
import cv2
from devices.hailo import Hailo
import GestureDetection as gd

#CAMERA
camera = Camera.Camera(resolution=(1280, 720))
camera.camera_open(correction=False)

#HAILO
hailo = Hailo("models/compiled/hand_landmark.hef")
model_h, model_w, _ = hailo.get_input_shape()
print(model_h, " x ", model_w)

#MODEL
hand_present = "hand_landmark/fc2"
handness = "hand_landmark/fc4"
normalized_locations = "hand_landmark/fc3"
world_locations = "hand_landmark/fc1"

#DETECTOR
detector = gd.GestureDetection(0.5)

while True:
	img = camera.frame
	if img is not None:
		#image processing
		frame = img.copy() 
		resized_frame = cv2.resize(frame, (model_w, model_h))
		#ai
		results = hailo.run(resized_frame)
		hand_presence = results[hand_present]
		hand_direction = results[handness]
		landmarks_world = results[world_locations]
		
		detector.update(hand_presence, hand_direction, landmarks_world)
		detector.draw_gesture(resized_frame, landmarks_world)
		print(detector.detected_gesture)
		
		cv2.imshow('frame', resized_frame)
		key = cv2.waitKey(1)
		if key == 27:
			break
			
hailo.close()
camera.camera_close()
cv2.destroyAllWindows()
