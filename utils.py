from sim_info import SimInfo
import vgamepad as vg
import time
import pyautogui

info = SimInfo()
old_progress = 0
laps = 0
gamepad = vg.VX360Gamepad()
gamepad.reset()
gamepad.update()

def parse_input():
    global old_progress
    # want to return 1d array with wheel slips, position, velocity, and normalized position.
    wheel_slips = list(info.physics.wheelSlip)
    velocity = list(info.physics.velocity)
    real_pos = list(info.graphics.carCoordinates)
    # 1d value tracking the distance along the spline. goes from 0-1
    progress = [info.graphics.normalizedCarPosition + 0]
    wheels_out = [info.physics.numberOfTyresOut]

    out = wheel_slips + velocity + real_pos + progress + wheels_out
    return out

def perform_output(inputs):
    global gamepad
    # inputs gas are from -1.0 to 1.0, constrain back to 0.0 to 1.0
    gas = abs((inputs[0]+1.0)/2.0)
    brake = abs((inputs[1]+1.0)/2.0)
    # steering format is fine as is.
    steering = inputs[2]

    gamepad.right_trigger_float(gas)
    gamepad.left_trigger_float(brake)
    gamepad.left_joystick_float(steering,0)
    gamepad.update()

def restart():
    global old_progress
    gamepad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_RIGHT_SHOULDER)
    gamepad.update()
    time.sleep(0.1)
    gamepad.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_RIGHT_SHOULDER)
    gamepad.update()
    time.sleep(0.1)
    old_progress = info.graphics.normalizedCarPosition
    laps = info.graphics.completedLaps

def step(action):
    perform_output(action)
    vals = parse_input()
    delta = info.graphics.normalizedCarPosition - old_progress
    done = False
    reward = delta
    if info.graphics.completedLaps > laps:
        done = True
        reward += 100

    if info.physics.numberOfTyresOut > 2:
        done = True
        reward -= 100

    return vals, reward, done