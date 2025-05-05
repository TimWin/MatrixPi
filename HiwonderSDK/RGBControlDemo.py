import time
import signal
import ros_robot_controller_sdk as rrc

print('''
**********************************************************
********功能:幻尔科技树莓派扩展板，RGB灯控制例程(function: Hiwonder Raspberry Pi expansion board, RGB light control routine)**********
**********************************************************
----------------------------------------------------------
Official website:https://www.hiwonder.com
Online mall:https://hiwonder.tmall.com
----------------------------------------------------------
Tips:
 * 按下Ctrl+C可关闭此次程序运行，若失败请多次尝试！(press Ctrl+C to close the running program, please try multiple times if failed)
----------------------------------------------------------
''')

start = True
#关闭前处理(process program before closing)
def Stop(signum, frame):
    global start

    start = False
    print('关闭中...')
board = rrc.Board()
#先将所有灯关闭(turn all lights off firstly)
board.set_rgb([[1, 0, 0, 0], [2, 0, 0, 0]])
signal.signal(signal.SIGINT, Stop)

while True:
    #设置2个灯为红色(set two lights to red)
    board.set_rgb([[1, 255, 0, 0], [2, 255, 0, 0]])
    time.sleep(1)
    
    #设置2个灯为绿色(set two lights to green)
    board.set_rgb([[1, 0, 255, 0], [2, 0, 255, 0]])
    time.sleep(1)
    
    #设置2个灯为蓝色(set two lights to blue)
    board.set_rgb([[1, 0, 0, 255], [2, 0, 0, 255]])
    time.sleep(1)
    
    #设置2个灯为黄色(set two lights to yellow)
    board.set_rgb([[1, 255, 255, 0], [2, 255, 255, 0]])
    time.sleep(1)

    if not start:
        #所有灯关闭(turn all lights off)
        board.set_rgb([[1, 0, 0, 0], [2, 0, 0, 0]])
        print('已关闭')
        break
