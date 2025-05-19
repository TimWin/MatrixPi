import Camera
import cv2
from devices.hailo import Hailo
from GestureDetection import *

import HiwonderSDK.PID as PID
import HiwonderSDK.mecanum as mecanum
import HiwonderSDK.ros_robot_controller_sdk as rrc

#ROBOT
board = rrc.Board()
def robot_move_forward(duty = 50):
	board.set_motor_duty([[1, duty], [2, -duty], [3, duty], [4, -duty]])

def robot_move_left(duty = 50):
	board.set_motor_duty([[1, duty], [2, duty], [3, duty], [4, duty]])
	
def robot_move_right(duty = 50):
	board.set_motor_duty([[1, -duty], [2, -duty], [3, -duty], [4, -duty]])
	
def robot_strafe_right(duty = 50):
	board.set_motor_duty([[1, -duty], [2, duty], [3, duty], [4, -duty]])
	
def robot_strafe_left(duty = 50):
	board.set_motor_duty([[1, duty], [2, -duty], [3, -duty], [4, duty]])

def robot_move_backward(duty = 50):
	robot_move_forward(-duty)
	
def robot_abort_move():
	robot_move_forward(0)

def move_based_on_gesture(gesture):
	match gesture:
		case Gestures.Unknown:
			robot_abort_move()
			#print("Abort Unknown")
		case Gestures.Fist:
			robot_abort_move()
			#print("Abort Fist")
		case Gestures.HandOpen:
			robot_move_forward()
			#print("HandOpen")
		case Gestures.HandClosed:
			robot_move_backward()
			#print("HandClosed")
		case Gestures.RotateRightOpen:
			robot_move_right()
			#print("RotateRightOpen")
		case Gestures.RotateRightClosed:
			robot_strafe_right()
			#print("RotateRoghtClosed")
		case Gestures.RotateLeftOpen:
			robot_move_left()
			#print("RotateLeftOpen")
		case Gestures.RotateLeftClosed:
			robot_strafe_left()
			#print("RotateLeftClosed")
	if gesture != Gestures.Unknown and gesture != Gestures.Fist:
		time.sleep(0.020) # sleep for some time for smoother operation
	#else:
	#	print("Discard")

# Servo
servo_x = 1500
servo_y = 1500
servo_x_pid = PID.PID(P=0.2, I=0.0005, D=0.0000) 
#servo_y_pid = PID.PID(P=0.2, I=0.000, D=0.000)
def track_based_on_center(gesture, center_point, width, height):
	global servo_x, servo_y
	
	if gesture == Gestures.Unknown or gesture == Gestures.Fist:
		return
	
	center_x = center_point[0]
	center_y = center_point[1]
	if center_x != 0 and center_y != 0:
		if abs(center_x - width/2.0) < 15: 
			center_x = width/2.0
		servo_x_pid.SetPoint = width/2.0
		servo_x_pid.update(center_x) 
		servo_x += int(servo_x_pid.output) 

		servo_x = 800 if servo_x < 800 else servo_x 
		servo_x = 2200 if servo_x > 2200 else servo_x
		
		#if abs(center_y - height/2.0) < 10: 
		#	center_y = height/2.0
		#servo_y_pid.SetPoint = height/2.0  
		#servo_y_pid.update(center_y)
		#servo_y -= int(servo_y_pid.output) 

		#servo_y = 1000 if servo_y < 1000 else servo_y 
		#servo_y = 1900 if servo_y > 1900 else servo_y 
		board.pwm_servo_set_position(0.02, [[1, servo_y], [2, servo_x]])  
		
#CAMERA
camera = Camera.Camera(resolution=(1280, 720))
camera.camera_open(correction=False)

#HAILO
hailo = Hailo("/home/pi/projects/git/New/MatrixPi/models/compiled/hand_landmark.hef")
model_h, model_w, _ = hailo.get_input_shape()
print(model_h, " x ", model_w)

#MODEL
hand_present = "hand_landmark/fc2"
handness = "hand_landmark/fc4"
normalized_locations = "hand_landmark/fc3"
world_locations = "hand_landmark/fc1"

#DETECTOR
detector = GestureDetection(0.75, buffer_size=24)
gesture_counter = 0
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
		#analyze
		detector.update(hand_presence, hand_direction, landmarks_world)
		detector.draw_gesture(resized_frame, landmarks_world)
		#move
		move_based_on_gesture(detector.detected_gesture)
		track_based_on_center(detector.detected_gesture, detector.get_hand_center(), model_w, model_h)
		
		#show :-)
		if gesture_counter > 16:
			print(detector.detected_gesture)
			gesture_counter = 0
		gesture_counter = gesture_counter + 1
		
		big_frame = cv2.resize(resized_frame, (model_w*2, model_h*2))
		
		cv2.imshow('frame', big_frame)
		key = cv2.waitKey(1)
		if key == 27:
			break
			
hailo.close()
camera.camera_close()
cv2.destroyAllWindows()
