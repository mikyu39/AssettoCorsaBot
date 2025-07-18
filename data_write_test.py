import vgamepad as vg
import time

gamepad = vg.VX360Gamepad()

gamepad.reset()
gamepad.right_trigger_float(0.5)

gamepad.update()
time.sleep(0.01)

while True:
    # time.sleep(0.01)
    # gamepad.left_joystick_float(-1, 0)
    # gamepad.update()
    # time.sleep(0.01)
    # gamepad.left_joystick_float(1, 0)
    # gamepad.update()
    gamepad.right_trigger_float(0.5)
    time.sleep(0.01)
    gamepad.update()
