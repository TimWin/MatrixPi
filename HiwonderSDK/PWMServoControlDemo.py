#!/usr/bin/python3
# coding=utf8
import sys
sys.path.append('/home/pi/TurboPi/')
import time
import signal
import threading
import ros_robot_controller_sdk as rrc

if sys.version_info.major == 2:
    print('Please run this program with python3!')
    sys.exit(0)
    
print('''
**********************************************************
********功能:幻尔科技树莓派扩展板，PWM舵机控制例程(function: Hiwonder Raspberry Pi expansion board, PWM servo control routine)**********
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
start = True
#关闭前处理(process program before closing)
def Stop(signum, frame):
    global start

    start = False
    print('关闭中...')

signal.signal(signal.SIGINT, Stop)

if __name__ == '__main__':
    
    while True:
        board.pwm_servo_set_position(1, [[1, 1100]]) # 设置1号舵机脉宽为2500，运行时间为1000毫秒(set servo 1 pulse width to 2500, running time to 1000 milliseconds)
        time.sleep(1)
        board.pwm_servo_set_position(1, [[1, 1500]]) # 设置1号舵机脉宽为1500，运行时间为1000毫秒(set servo 1 pulse width to 1500, running time to 1000 milliseconds)
        time.sleep(1)
        board.pwm_servo_set_position(1, [[1, 1900], [2, 1000]]) #设置1号舵机脉宽为1000，设置2号舵机脉宽为1000, 运行时间为1000毫秒(set servo 1 pulse width to 1000, set servo 2 pulse width to 1000)
        time.sleep(1)
        board.pwm_servo_set_position(1, [[1, 1500], [2, 2000]]) #设置1号舵机脉宽为2000，设置2号舵机脉宽为2000, 运行时间为1000毫秒(set servo 1 pulse width to 2000, set servo 2 pulse width to 2000, running time to 1000 milliseconds)
        time.sleep(1)       
        if not start:
            board.pwm_servo_set_position(1, [[1, 1500], [2, 1500]]) # 设置1号舵机脉宽为1500，设置2号舵机脉宽为1500, 运行时间为1000毫秒(set servo 1 pulse width to 1500, set servo 2 pulse width to 1500, running time to 1000 milliseconds)
            time.sleep(1)
            print('已关闭')
            break
    
    
        
