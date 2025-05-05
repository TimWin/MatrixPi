#!/usr/bin/python3
# coding=utf8
import sys
sys.path.append('/home/pi/TurboPi/')
import time
import signal
import ros_robot_controller_sdk as rrc

if sys.version_info.major == 2:
    print('Please run this program with python3!')
    sys.exit(0)
    
print('''
**********************************************************
********功能:幻尔科技树莓派扩展板，电机控制例程(function: Hiwonder Raspberry Pi expansion board, motor control routine)**********
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
    board.set_motor_duty([[1, 0], [2, 0], [3, 0], [4, 0]])  # 关闭所有电机(close all motors)

signal.signal(signal.SIGINT, Stop)

if __name__ == '__main__':
    
    while True:
        board.set_motor_duty([[1, 35]])  #设置1号电机速度35(set motor 1 speed to 35)
        time.sleep(0.2)
        board.set_motor_duty([[1, 90]])  #设置1号电机速度90(set motor 1 speed to 90)
        time.sleep(0.2)    
        
        if not start:
            board.set_motor_duty([[1, 0], [2, 0], [3, 0], [4, 0]])  # 关闭所有电机(close all motors)
            print('已关闭')
            break
    
    
        
