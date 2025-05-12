import cv2
import Camera
import time
import math
import numpy as np
import threading

# Camera test
camera = Camera.Camera(resolution=(1280, 720))
camera.camera_open(correction=False)
img = camera.frame

gst_str = (
    "appsrc device=/dev/video0 ! video/x-raw,width=640,height=480,framerate=30/1 ! videoconvert ! x264enc tune=zerolatency key-int-max=15 intra-refresh=true speed-preset=superfast ! rtph264pay config-interval=1 pt=96 ! udpsink host=192.168.0.20 port=5000"
) # Replace with actual IP

# Create GStreamer video writer
gst_out = cv2.VideoWriter(
    gst_str,
    cv2.CAP_GSTREAMER,
    0,  # fourcc (not used)
    30,  # fps
    (640, 480),  # frame size
    True,  # isColor
)

if not gst_out.isOpened():
    print("Failed to open GStreamer pipeline.")
    exit(1)
    
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
		gst_out.write(frame)

gst_out.release()
cv2.destroyAllWindows()
