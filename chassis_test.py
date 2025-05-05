import HiwonderSDK.PID as PID
import HiwonderSDK.mecanum as mecanum
import HiwonderSDK.ros_robot_controller_sdk as rrc

board = rrc.Board()
print(board.get_battery())
