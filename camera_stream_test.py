import cv2
import Camera
import time
import math
import numpy as np
import threading
from flask import Flask, render_template, Response

# Camera test
camera = Camera.Camera(resolution=(1280, 720))
camera.camera_open(correction=False)
img = camera.frame

app = Flask(__name__)

#gst_str = (
#    "appsrc device=/dev/video0 ! video/x-raw,width=640,height=480,framerate=30/1 ! videoconvert ! x264enc tune=zerolatency key-int-max=15 intra-refresh=true speed-preset=superfast ! rtph264pay config-interval=1 pt=96 ! udpsink host=192.168.0.20 port=5000"
#) # Replace with actual IP

# Create GStreamer video writer
#gst_out = cv2.VideoWriter(
#    gst_str,
#    cv2.CAP_GSTREAMER,
#    0,  # fourcc (not used)
#    30,  # fps
#    (640, 480),  # frame size
#    True,  # isColor
#)

#if not gst_out.isOpened():
#    print("Failed to open GStreamer pipeline.")
#    exit(1)

def generate_frames():
    while True:
        img = camera.get_next_frame_blocking()
        if img is not None:
            camera_frame=img.copy()
            result, frame_buffer = cv2.imencode(".jpg", camera_frame)
            frame = frame_buffer.tobytes()
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
	    
@app.route('/')
def index():
    return "<h1>Camera Streaming</h1><img src='/video'>"

@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
    
#main
#prev_time = time.perf_counter()
#while not finish_process:
#	current_time = time.perf_counter()
#	elapsed_time = (current_time - prev_time) * 1000.0
#	prev_time = current_time
#	print(f"Frame processing took {elapsed_time:.6f} ms")
#	
#	img = camera.frame
#	if img is not None:
#		frame = img.copy()
#		gst_out.write(frame)

cv2.destroyAllWindows()
