import cv2
import threading
from flask import Flask, render_template, Response
import time

class FrameStreamer:
    def __init__(self, quality=80):
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        self.frame_ready = threading.Condition(lock=self.frame_lock)
        self.last_sent_time = time.time()
        self.jpeg_quality = quality

        # Flask app will be created inside the class
        self.app = FrameStreamer._create_app(self)

    # Static method to create the Flask app
    @staticmethod
    def _create_app(instance):
        app = Flask(__name__)
        app.add_url_rule('/', 'index', instance.index)
        app.add_url_rule('/video', 'video', instance.video)
        return app

    # External method to pass frame data
    # Add a queue for buffering
    def update_frame(self, frame):
        with self.frame_ready:
            self.latest_frame = frame.copy()
            self.frame_ready.notify_all()

    # Stream the latest frame in MJPEG format
    def _generate_frames(self):
        while True:
            with self.frame_ready:
                #if self.latest_frame is None:
                self.frame_ready.wait()  # Wait until frame is available

            # Encode frame into JPEG and yield it
            ret, frame_buffer = cv2.imencode('.jpg', self.latest_frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
            frame = frame_buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    # Route: Serve the index page
    def index(self):
        return "<h1>Camera Stream</h1><img src='/video'>"

    # Route: Serve the video stream
    def video(self):
        return Response(self._generate_frames(),
                        mimetype='multipart/x-mixed-replace; boundary=frame')


# Function to simulate external camera capture thread
def capture_frames(streamer):
    cap = cv2.VideoCapture(0)  # Open camera
    while True:
        success, frame = cap.read()
        if success:
            streamer.update_frame(frame)  # Pass frame to the streamer


if __name__ == '__main__':
    # Instantiate the camera streamer and start the capture thread
    streamer = FrameStreamer()
    threading.Thread(target=capture_frames, args=(streamer,), daemon=True).start()

    # Start Flask app (the user doesn't need to know Flask is involved)
    streamer.app.run(host='0.0.0.0', port=5000, threaded=True)
