import time
import ros_robot_controller_sdk as rrc

print('''
**********************************************************
********功能:幻尔科技树莓派扩展板，蜂鸣器控制例程(function: Hiwonder Raspberry Pi expansion board, buzzer control routine)*********
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
board.set_buzzer(1900, 0.1, 0.9, 1) # 以1900Hz的频率，持续响0.1秒，关闭0.9秒，重复1次(at a frequency of 1900Hz, continuously sound for 0.1 seconds, then silence for 0.9 seconds, repeat once)
time.sleep(2)
board.set_buzzer(1000, 0.5, 0.5, 0) # 以1000Hz的频率，持续响0.5秒，关闭0.5秒，一直重复(at a frequency of 1000Hz, continuously sound for 0.5 seconds, then silence for 0.5 seconds, repeat once)
time.sleep(3)
board.set_buzzer(1000, 0.0, 0.0, 1) # 关闭(close)
