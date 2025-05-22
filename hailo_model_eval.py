import Camera
import cv2
from devices.hailo import Hailo
from GestureDetection import *
import time
from Video.FrameStream import FrameStreamer

#CAMERA
camera = Camera.Camera(resolution=(1280, 720))
camera.camera_open(correction=False)

#HAILO
hailo = Hailo("/home/pi/projects/git/New/MatrixPi/models/compiled/hand_landmark.hef")
model_h, model_w, _ = hailo.get_input_shape()
print(model_h, " x ", model_w)

#MODEL
#hand_present = "hand_landmark_full/fc4"
#handness = "hand_landmark_full/fc3"
#normalized_locations = "hand_landmark_full/fc2"
#world_locations = "hand_landmark_full/fc1"

hand_present = "hand_landmark/fc2"
handness = "hand_landmark/fc4"
normalized_locations = "hand_landmark/fc3"
world_locations = "hand_landmark/fc1"

#DETECTOR
detector = GestureDetection(0.15, buffer_size=24)

#STREAM
frame_streamer = FrameStreamer()
frame_streamer.start(host='0.0.0.0', port=5000)

#FILTER
kalman = cv2.KalmanFilter(4, 2)  # 4 state variables (x, y, dx, dy), 2 measurements (x, y)
dt = 1.0 / 30.0  # Assuming 30 FPS for video capture

# Set up Kalman filter transition matrix (constant velocity model)
num_landmarks = 21
kalman_filters = [cv2.KalmanFilter(4, 2) for _ in range(num_landmarks)]
for f in range(num_landmarks):
	kalman_filters[f].transitionMatrix = np.array([
		[1, 0, dt, 0],   # x = x + dx * dt
		[0, 1, 0, dt],   # y = y + dy * dt
		[0, 0, 1, 0],    # dx = dx
		[0, 0, 0, 1],    # dy = dy
	], dtype=np.float32)

	# Set up measurement matrix (only position is observed)
	kalman_filters[f].measurementMatrix = np.array([
		[1, 0, 0, 0],
		[0, 1, 0, 0]
	], dtype=np.float32)

	kalman_filters[f].processNoiseCov = np.eye(4, dtype=np.float32) * 0.1
	kalman_filters[f].measurementNoiseCov = np.eye(2, dtype=np.float32) * 1
	
def draw_locations(img, locations, color=(0,0,255)):
	reshaped_int_locations = [[int(locations[i]), int(locations[i+1])] for i in range(0, len(locations), 3)]
	for loc in reshaped_int_locations:
		cv2.circle(img, loc, 6, color, thickness=1)

def update_kalman(kalman, x, y):
    """Update Kalman filter with a new measurement (x, y)"""
    prediction = kalman.predict()
    measurement = np.array([[x], [y]])
    corrected = kalman.correct(measurement)
    return corrected #corrected[0, 0], corrected[1, 0] 

while True:
	img = camera.get_next_frame_blocking()
	if img is not None:
		#image processing
		frame = img.copy() 
		resized_frame = cv2.resize(frame, (model_w, model_h))
		converted_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
		#resized_frame = resized_frame.astype(np.float32) / 255.0
		#ai
		results = hailo.run(converted_frame)
		locations_world = results[world_locations]
		presence_estimation = results[hand_present]
		
		#filter locations
		locations_filter = locations_world.copy()
		for i in range(num_landmarks):
			state = update_kalman(kalman_filters[i], locations_world[i*3+0], locations_world[i*3+1])
			locations_filter[i*3] = state[0]
			locations_filter[i*3+1] = state[1]
			#if i == 8:
			#	print(state[2], "x", state[3])
		
		if presence_estimation > 0.65:
			#x_filter, y_filter = update_kalman(kalman_filters[8], locations_world[8*3+0], locations_world[8*3+1])
			#print(x_filter, " ", locations_world[8*3+0], " | ", y_filter, " ", locations_world[8*3+1])
			draw_locations(resized_frame, locations_world)
		draw_locations(resized_frame, locations_filter, (0, 255, 0))
			#break
		
		
		#analyze
		#detector.update(results[hand_present], results[handness], results[world_locations])
		#detector.draw_gesture(resized_frame, results[world_locations])
		
		stream_frame = cv2.resize(resized_frame, (model_w*4, model_h*4))
		frame_streamer.update_frame(stream_frame)
		#time.sleep(0.050)
		
hailo.close()
