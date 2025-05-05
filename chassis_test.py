import HiwonderSDK.PID as PID
import HiwonderSDK.mecanum as mecanum
import HiwonderSDK.ros_robot_controller_sdk as rrc
import time

board = rrc.Board()

def move_forward(duty = 50):
	board.set_motor_duty([[1, duty], [2, -duty], [3, duty], [4, -duty]])

def move_left(duty = 50):
	board.set_motor_duty([[1, duty], [2, duty], [3, duty], [4, duty]])
	
def move_right(duty = 50):
	board.set_motor_duty([[1, -duty], [2, -duty], [3, -duty], [4, -duty]])

def move_backward(duty = 50):
	move_forward(-duty)
	
def abort_move():
	move_forward(0)

move_forward()
# board.set_motor_speed([[1, -0.3], [2, 0.3], [3, -0.3], [4, 0.3]])
time.sleep(0.5)
move_left()
time.sleep(0.5)
move_right()
time.sleep(0.5)
abort_move()
time.sleep(0.5)
move_backward()
time.sleep(0.5)
# board.set_rgb([[1, 0, 0, 255]])
abort_move()
