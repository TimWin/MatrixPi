import argparse
import cv2
import Camera
import time
import math
import numpy as np
from enum import Enum
from devices.hailo import Hailo
from collections import deque, Counter

import HiwonderSDK.PID as PID
import HiwonderSDK.mecanum as mecanum
import HiwonderSDK.ros_robot_controller_sdk as rrc
#robot
board = rrc.Board()
def robot_move_forward(duty = 50):
	board.set_motor_duty([[1, duty], [2, -duty], [3, duty], [4, -duty]])

def robot_move_left(duty = 50):
	board.set_motor_duty([[1, duty], [2, duty], [3, duty], [4, duty]])
	
def robot_move_right(duty = 50):
	board.set_motor_duty([[1, -duty], [2, -duty], [3, -duty], [4, -duty]])

def robot_move_backward(duty = 50):
	robot_move_forward(-duty)
	
def robot_abort_move():
	robot_move_forward(0)

# Camera test
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

#FINGER
thumb_indices = (0,2,3,4) # debug
index_indices = (5,6,7,8)
middle_indices = (9,10,11,12)
ring_indices = (13,14,15,16)
pinky_indices = (17,18,19,20)

#Gestures
Gestures = Enum("Gestures", "Unknown Fist HandOpen HandClosed")
current_gesture = Gestures.Unknown
gesture_center_x = -1
gesture_center_y = -1

# Servo
servo_x = 1500
servo_y = 1500
servo_x_pid = PID.PID(P=0.1, I=0.0000, D=0.0000) 
servo_y_pid = PID.PID(P=0.0125, I=0.0005, D=0.000)

#Circular Buffer
# Create a circular buffer of size 16
gesture_buffer = deque(maxlen=16)
# Function to add to the buffer
def add_to_buffer(value: Gestures):
    gesture_buffer.append(value)
    
#LED control
def set_led(color):
    if color == "red":
        board.set_rgb([[1, 255, 0, 0], [2, 255, 0, 0]])
    elif color == "green":
        board.set_rgb([[1, 0, 255, 0], [2, 0, 255, 0]])
    elif color == "blue":
        board.set_rgb([[1, 0, 0, 255], [2, 0, 0, 255]])
    elif color == "yellow":
        board.set_rgb([[1, 0, 255, 255], [2, 0, 255, 255]])
    else:
        board.set_rgb([[1, 0, 0, 0], [2, 0, 0, 0]])

def calc_area(img, reshaped_locations):
	global gesture_center_x
	global gesture_center_y
	
	x_coords = reshaped_locations[:,0]
	y_coords = reshaped_locations[:,1]
	min_x, max_x = x_coords.min(), x_coords.max()
	min_y, max_y = y_coords.min(), y_coords.max()
	cv2.rectangle(img, (int(min_x), int(min_y)), (int(max_x), int(max_y)), (0,255,0), 2)
	area = (max_y - min_y) * (max_x - min_x)
	img_h, img_w = img.shape[:2]
	gesture_center_x =  int(min_x + max_x) / 2
	gesture_center_y =  int(min_y + max_y) / 2
	#print(gesture_center_x, ", ", gesture_center_y)
	#print(area)
	#print("(", min_x, ",", min_y, "), (", max_x, ",", max_y, ")")
	
def angle_between_vectors(v1, v2):
	v1_u = v1 / np.linalg.norm(v1)
	v2_u = v2 / np.linalg.norm(v2)
	dot = np.dot(v1_u, v2_u)
	clipped_dot = np.clip(dot, -1.0, 1.0)
	return np.degrees(np.arccos(clipped_dot))
	
def calc_finger_angles(reshaped_locations, indices = (5,6,7,8)):
	lower_vector = reshaped_locations[indices[1]] - reshaped_locations[indices[0]]
	middle_vector = reshaped_locations[indices[2]] - reshaped_locations[indices[1]]
	upper_vector = reshaped_locations[indices[3]] - reshaped_locations[indices[2]]
	lower_angle = angle_between_vectors(lower_vector, middle_vector)
	upper_angle = angle_between_vectors(middle_vector, upper_vector)
	#print("index angles: ", lower_angle, " ", upper_angle)
	return (lower_angle, upper_angle)
	
def is_finger_straight(reshaped_locations, finger_indices, angle_threshold = 15):
	angles = calc_finger_angles(reshaped_locations, finger_indices)
	straight = all(angle < angle_threshold for angle in angles)
	return straight
	
def is_hand_openend(reshaped_locations, angle_threshold = 15):
	#thumb = is_finger_straight(reshaped_locations, thumb_indices, angle_threshold)
	index = is_finger_straight(reshaped_locations, index_indices, angle_threshold)
	middle = is_finger_straight(reshaped_locations, middle_indices, angle_threshold)
	ring = is_finger_straight(reshaped_locations, ring_indices, angle_threshold)
	pinky = is_finger_straight(reshaped_locations, pinky_indices, angle_threshold)
	return index and middle and ring and pinky
	
def gesture_index_only(reshaped_locations, angle_threshold=15):
	index = is_finger_straight(reshaped_locations, index_indices, angle_threshold)
	middle = is_finger_straight(reshaped_locations, middle_indices, angle_threshold)
	ring = is_finger_straight(reshaped_locations, ring_indices, angle_threshold)
	pinky = is_finger_straight(reshaped_locations, pinky_indices, angle_threshold)
	#print(index, ",", middle, ",", ring, ",", pinky)
	return index and not middle and not ring and not pinky
	
def calc_angles_between_fingers(reshaped_locations):
    index_vector = reshaped_locations[8] - reshaped_locations[5]
    middle_vector = reshaped_locations[12] - reshaped_locations[9]
    ring_vector = reshaped_locations[16] - reshaped_locations[13]
    pinky_vector = reshaped_locations[20] - reshaped_locations[17]
    return ( angle_between_vectors(index_vector, middle_vector), angle_between_vectors(middle_vector, ring_vector), angle_between_vectors(ring_vector, pinky_vector) )
	
def draw_finger(img, locations, indices):
	center = locations[indices[0]]
	middle = locations[indices[1]]
	tip = locations[indices[2]]
	cv2.line(img, center, middle, (0,0,255), 3)
	cv2.line(img, middle, tip, (0,0,255), 3)
	
def draw_fingers(img, locations):
	#int_locations = [int(x) for x in locations]
	reshaped_int_locations = [[int(locations[i]), int(locations[i+1])]
                     for i in range(0, len(locations), 3)]
        #thumb
	draw_finger(img, reshaped_int_locations, (0, 2, 4))
	#index
	draw_finger(img, reshaped_int_locations, (5, 7, 8))
	#middle
	draw_finger(img, reshaped_int_locations, (9, 11, 12))
	#ring
	draw_finger(img, reshaped_int_locations, (13, 15, 16))
	#pinky
	draw_finger(img, reshaped_int_locations, (17, 19, 20))

def analyze_model_results(img, results, presence_threshold=0.5):
    global current_gesture
    current_gesture = Gestures.Unknown
    if results is not None:
        p_present = results[hand_present]
        if p_present> presence_threshold:
            #convert normalized to pixel
            width = img.shape[1]
            height = img.shape[0]
            locations = results[world_locations]
            reshaped_locations = locations.reshape(-1,3)
            calc_area(img, reshaped_locations)
	    
            hand_opened = is_hand_openend(reshaped_locations, 20)
            if hand_opened:
                angles = calc_angles_between_fingers(reshaped_locations)
                fingers_closed = all([x < 10 for x in angles])
                if fingers_closed:
                    current_gesture = Gestures.HandClosed
                else:
                    current_gesture = Gestures.HandOpen
            else:
                current_gesture = Gestures.Fist
            
            draw_fingers(img, locations)

def move_based_on_gesture():
    # get most frequent item in buffer
    if len(gesture_buffer) < 16:
        return
    gesture_count = Counter(gesture_buffer)
    most_common_gesture = gesture_count.most_common(1)
    #print("Gesture: ", most_common_gesture[0][0])
    
    if most_common_gesture == Gestures.Unknown:
        robot_abort_move()
        set_led("yellow")
    elif most_common_gesture == Gestures.Fist:
        robot_abort_move()
        set_led("blue")
    elif most_common_gesture == Gestures.HandOpen:
        robot_move_forward()
        set_led("green")
    elif most_common_gesture == Gestures.HandClosed:
        robot_move_backward()
        set_led("blue")
	
def track_hand_with_servo(width, height):
    global gesture_center_x
    global gesture_center_y
    global servo_x, servo_y
    if gesture_center_x != -1 and gesture_center_y != -1:
        if abs(gesture_center_x - width/2.0) < 15: 
            gesture_center_x = width/2.0
        servo_x_pid.SetPoint = width/2.0 # 设定(set)
        servo_x_pid.update(gesture_center_x)     # 当前(current)
        servo_x += int(servo_x_pid.output)  # 获取PID输出值(get PID output value)
	
        servo_x = 800 if servo_x < 800 else servo_x # 设置舵机范围(set servo range)
        servo_x = 2200 if servo_x > 2200 else servo_x
	
	# 根据摄像头Y轴坐标追踪(track based on the camera Y-axis coordinates)
        if abs(gesture_center_y - height/2.0) < 10: # 移动幅度比较小，则不需要动(if the movement amplitude is small, no action is required)
            gesture_center_y = height/2.0
        servo_y_pid.SetPoint = height/2.0  
        servo_y_pid.update(gesture_center_y)
        servo_y -= int(servo_y_pid.output) # 获取PID输出值(gei PID output value)
	
        servo_y = 1000 if servo_y < 1000 else servo_y # 设置舵机范围(set servo range)
        servo_y = 1900 if servo_y > 1900 else servo_y
	# print(servo_y, center_y) 
	board.pwm_servo_set_position(0.02, [[1, servo_y], [2, servo_x]])  # 设置舵机移动(set servo movement)
        #print(servo_x, ", ", servo_y, ", ", servo_x_pid.output, ", ", servo_y_pid.output)
	
	
write_img_count = 0
while True:
    img = camera.frame
    if img is not None:
        frame = img.copy() 
        #imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
        resized_frame = cv2.resize(frame, (model_w, model_h))
        #start_time = time.perf_counter()
        results = hailo.run(resized_frame)
        #end_time = time.perf_counter()
        #elapsed_time = (end_time - start_time) * 1000.0
        #print(f"AI took {elapsed_time:.6f} ms")
        analyze_model_results(resized_frame, results)
        add_to_buffer(current_gesture)
        #print(current_gesture)
        track_hand_with_servo(model_w, model_h)
        move_based_on_gesture()

        cv2.imshow('frame', resized_frame)
        key = cv2.waitKey(1)
        if key == 27:
	        break
        elif key == 115:
            #print("TEST")
            path = "screenshots/shot_" + str(write_img_count) + ".png"
            cv2.imwrite(path, resized_frame)
            write_img_count = write_img_count + 1
			
    #time.sleep(0.03333)
hailo.close()
camera.camera_close()
cv2.destroyAllWindows()
