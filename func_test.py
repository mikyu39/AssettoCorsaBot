import vgamepad as vg
import time



print("Episode : {} \t\t Timestep : {} \t\t Average Reward : {:.3f}".format(1, 2, 3.0))


gamepad = vg.VX360Gamepad()

gamepad.update()

while True:
    gamepad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_RIGHT_SHOULDER)
    gamepad.update()
    time.sleep(0.1)
    gamepad.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_RIGHT_SHOULDER)
    gamepad.update()
    time.sleep(0.1)