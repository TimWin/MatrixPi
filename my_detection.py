import argparse
import cv2
import Camera
import time
import math
import numpy as np
from devices.hailo import Hailo

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

def calc_area(img, reshaped_locations):
	x_coords = reshaped_locations[:,0]
	y_coords = reshaped_locations[:,1]
	min_x, max_x = x_coords.min(), x_coords.max()
	min_y, max_y = y_coords.min(), y_coords.max()
	cv2.rectangle(img, (int(min_x), int(min_y)), (int(max_x), int(max_y)), (0,255,0), 2)
	area = (max_y - min_y) * (max_x - min_x)
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
    if results is not None:
        p_present = results[hand_present]
        if p_present> presence_threshold:
            #convert normalized to pixel
            width = img.shape[1]
            height = img.shape[0]
            locations = results[world_locations]
            reshaped_locations = locations.reshape(-1,3)
            calc_area(img, reshaped_locations)
            
            #angles = calc_finger_angles(reshaped_locations, middle_indices)
            #straight = all(angle < 15 for angle in angles)
            #print(straight)
            
            draw_fingers(img, locations)
            
            #angles = vector_2d_angle(locations)
            #thumb test
            #center = ( int(locations[0]), int(locations[1]) )
            #tip = ( int(locations[4*3+0]), int(locations[4*3+1]) )
            
            #index_start = ( int(locations[5*3+0]), int(locations[5*3+1]) )
            #index_end = ( int(locations[8*3+0]), int(locations[8*3+1]) )
            #middle_start = ( int(locations[9*3+0]), int(locations[9*3+1]) )
            #middle_end = ( int(locations[12*3+0]), int(locations[12*3+1]) )
            
            #cv2.line(img, center, tip, (255,0,0), 3)
            #cv2.line(img, index_start, index_end, (255,0,0), 3)
            #cv2.line(img, middle_start, middle_end, (255,0,0), 3)
            #print(reshaped_locations)

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
            #for key, value in results.items():
            #    print(key, ", ", value.shape)
            #break
            #print(results["hand_landmark/fc4"])
            #print(results[normalized_locations])
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
