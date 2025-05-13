import cv2
import time
import math
import numpy as np
from enum import Enum
from collections import deque, Counter

#Gestures
Gestures = Enum("Gestures", "Unknown Fist HandOpen HandClosed RotateRightOpen RotateRightClosed RotateLeftOpen RotateLeftClosed")

#FINGER
thumb_indices = (0,2,3,4) # debug
index_indices = (5,6,7,8)
middle_indices = (9,10,11,12)
ring_indices = (13,14,15,16)
pinky_indices = (17,18,19,20)

#enum? thumb ?
index_vector_index = 0
middle_vector_index = 1
ring_vector_index = 2
pinky_vector_index = 3

def angle_between_vectors(v1, v2):
	v1_u = v1 / np.linalg.norm(v1)
	v2_u = v2 / np.linalg.norm(v2)
	dot = np.dot(v1_u, v2_u)
	clipped_dot = np.clip(dot, -1.0, 1.0)
	return np.degrees(np.arccos(clipped_dot))


class GestureDetection:
	def __init__(self, presence_threshold, buffer_size=32, hand_opened_angle=25, fingers_closed_angle=10, hand_rotation_angle=20):
		self._current_gesture = Gestures.Unknown
		self._detected_gesture = Gestures.Unknown
		self._presence_threshold = presence_threshold
		self._gesture_buffer = deque(maxlen=buffer_size)
		self._hand_opened_angle = hand_opened_angle
		self._fingers_closed_angle = fingers_closed_angle
		self._hand_rotation_angle = hand_rotation_angle
		self._min_x = 0
		self._min_y = 0
		self._max_x = 0
		self._max_y = 0
		
	@property
	def detected_gesture(self):
		return self._detected_gesture
		
	def get_hand_center(self):
		center_x = int(self._min_x + self._max_x) / 2
		center_y = int(self._min_y + self._max_y) / 2
		return center_x, center_y
		
	def _calc_finger_angles(self, reshaped_locations, indices = (5,6,7,8)):
		lower_vector = reshaped_locations[indices[1]] - reshaped_locations[indices[0]]
		middle_vector = reshaped_locations[indices[2]] - reshaped_locations[indices[1]]
		upper_vector = reshaped_locations[indices[3]] - reshaped_locations[indices[2]]
		lower_angle = angle_between_vectors(lower_vector, middle_vector)
		upper_angle = angle_between_vectors(middle_vector, upper_vector)
		return (lower_angle, upper_angle)
		
	def _is_finger_straight(self, reshaped_locations, finger_indices):
		angles = self._calc_finger_angles(reshaped_locations, finger_indices)
		straight = all(angle < self._hand_opened_angle for angle in angles)
		return straight
		
	def _is_hand_openend(self, reshaped_locations):
		index = self._is_finger_straight(reshaped_locations, index_indices)
		middle = self._is_finger_straight(reshaped_locations, middle_indices)
		ring = self._is_finger_straight(reshaped_locations, ring_indices)
		pinky = self._is_finger_straight(reshaped_locations, pinky_indices)
		return index and middle and ring and pinky
		
	def _calc_finger_vectors(self, reshaped_locations):
		index_vector = reshaped_locations[8] - reshaped_locations[5]
		middle_vector = reshaped_locations[12] - reshaped_locations[9]
		ring_vector = reshaped_locations[16] - reshaped_locations[13]
		pinky_vector = reshaped_locations[20] - reshaped_locations[17]
		return (index_vector, middle_vector, ring_vector, pinky_vector)
	
	def _calc_angles_between_fingers(self, finger_vectors):
		return ( angle_between_vectors(finger_vectors[index_vector_index], finger_vectors[middle_vector_index]), angle_between_vectors(finger_vectors[middle_vector_index], finger_vectors[ring_vector_index]), angle_between_vectors(finger_vectors[ring_vector_index], finger_vectors[pinky_vector_index]) )
	
	def _calc_hand_angle(self, finger_vectors):
		angle = 90 - angle_between_vectors(finger_vectors[middle_vector_index], (1,0,0) )
		return angle
		
	def _calc_frame(self, reshaped_locations):
		x_coords = reshaped_locations[:,0]
		y_coords = reshaped_locations[:,1]
		self._min_x, self._max_x = x_coords.min(), x_coords.max()
		self._min_y, self._max_y = y_coords.min(), y_coords.max()
	
	def _compute_gesture(self):
		gesture_count = Counter(self._gesture_buffer)
		self._detected_gesture = gesture_count.most_common(1)[0][0]
	
	def update(self, hand_present, hand_direction, world_locations):
		self.current_gesture = Gestures.Unknown
		if hand_present > self._presence_threshold:
			reshaped_locations = world_locations.reshape(-1,3)
			hand_opened = self._is_hand_openend(reshaped_locations)
			if hand_opened:
				#calc parameters
				self._calc_frame(reshaped_locations)
				finger_vectors = self._calc_finger_vectors(reshaped_locations)
				finger_angles = self._calc_angles_between_fingers(finger_vectors)
				hand_angle = self._calc_hand_angle(finger_vectors)
				fingers_closed = all([x < self._fingers_closed_angle for x in finger_angles])
				
				#analyze
				tilted_left = False
				tilted_right = False
				if abs(hand_angle) > self._hand_rotation_angle:
					if hand_angle > 0:
						tilted_right = True
					else:
						tilted_left = True
					
				if fingers_closed:
					if tilted_right:
						self._current_gesture = Gestures.RotateRightClosed
					elif tilted_left:
						self._current_gesture = Gestures.RotateLeftClosed
					else:
						self._current_gesture = Gestures.HandClosed
				else:
					if tilted_right:
						self._current_gesture = Gestures.RotateRightOpen
					elif tilted_left:
						self._current_gesture = Gestures.RotateLeftOpen
					else:
						self._current_gesture = Gestures.HandOpen
			else:
				self._current_gesture = Gestures.Fist
		
		self._gesture_buffer.append(self._current_gesture)
		self._compute_gesture()
		
	def _draw_finger(self, img, locations, indices):
		center = locations[indices[0]]
		middle = locations[indices[1]]
		tip = locations[indices[2]]
		cv2.line(img, center, middle, (0,0,255), 3)
		cv2.line(img, middle, tip, (0,0,255), 3)
		
	def draw_gesture(self, img, locations):
		cv2.rectangle(img, (int(self._min_x), int(self._min_y)), (int(self._max_x), int(self._max_y)), (0,255,0), 2)
		reshaped_int_locations = [[int(locations[i]), int(locations[i+1])]
                     for i in range(0, len(locations), 3)]
        #thumb
		self._draw_finger(img, reshaped_int_locations, (0, 2, 4))
		#index
		self._draw_finger(img, reshaped_int_locations, (5, 7, 8))
		#middle
		self._draw_finger(img, reshaped_int_locations, (9, 11, 12))
		#ring
		self._draw_finger(img, reshaped_int_locations, (13, 15, 16))
		#pinky
		self._draw_finger(img, reshaped_int_locations, (17, 19, 20))
