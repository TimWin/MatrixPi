#!/usr/bin/python3
# coding=utf8
import sys
import time
sys.path.append('/home/pi/TurboPi/')
import HiwonderSDK.ros_robot_controller_sdk as rrc

if sys.version_info.major == 2:
    print('Please run this program with python3!')
    sys.exit(0)
    
print('''
**********************************************************
********************PWM舵机和电机测试(PWM servo and motor test)************************
**********************************************************
----------------------------------------------------------
Official website:https://www.hiwonder.com
Online mall:https://hiwonder.tmall.com
----------------------------------------------------------
Tips:
 * 按下Ctrl+C可关闭此次程序运行，若失败请多次尝试！(press Ctrl+C to close the running program, please try multiple times if failed)
----------------------------------------------------------
''')
board = rrc.Board()
board.pwm_servo_set_position(0.3, [[1, 1800]]) 
time.sleep(0.3)
board.pwm_servo_set_position(0.3, [[1, 1500]]) 
time.sleep(0.3)
board.pwm_servo_set_position(0.3, [[1, 1200]]) 
time.sleep(0.3)
board.pwm_servo_set_position(0.3, [[1, 1500]]) 
time.sleep(1.5)

board.pwm_servo_set_position(0.3, [[2, 1200]]) 
time.sleep(0.3)
board.pwm_servo_set_position(0.3, [[2, 1500]]) 
time.sleep(0.3)
board.pwm_servo_set_position(0.3, [[2, 1800]])
time.sleep(0.3)
board.pwm_servo_set_position(0.3, [[2, 1500]]) 
time.sleep(1.5)

board.set_motor_duty([[1, -45]])
time.sleep(0.5)
board.set_motor_duty([[1, 0]])
time.sleep(1)
        
board.set_motor_duty([[2, 45]])
time.sleep(0.5)
board.set_motor_duty([[2, 0]])
time.sleep(1)

board.set_motor_duty([[3, -45]])
time.sleep(0.5)
board.set_motor_duty([[3, 0]])
time.sleep(1)

board.set_motor_duty([[4, 45]])
time.sleep(0.5)
board.set_motor_duty([[4, 0]])
time.sleep(1)

