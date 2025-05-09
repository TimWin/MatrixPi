# Hand gesture controlled TurboPI with PI AI HAT

## Note: Everything is WIP

## Repo Contents
- CameraCalibtration (From TurboPi SDK):
  - utility functions for camera correction
  - not to be used when webcam is main device
- HiWonderSDK (From TurboPi SDK):
  - functions for TurboPi control
  - e.g. motor control commands
- devices (from picamera2):
  - hailo device control
- models:
  - trained models for the HAILO8L device (AI HAT)
- screenshots:
  - the application can take a screenshot with the keyboard key 's'
- buffer_test.py:
  - test script for the circular buffer
- chassis_test.py:
  - test script for HiWonderSDK functions
- my_detections.py:
  - main application
  - exit with escape
