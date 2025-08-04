from sim_info import SimInfo
import vgamepad as vg
import time
import numpy as np

info = SimInfo()
old_progress = 0
iters_not_moving = 0


laps = 0
gamepad = vg.VX360Gamepad()
gamepad.reset()
gamepad.update()

SPEED_MULT = 0.5
NOT_MOVING_THRESHOLD = 10000
LAP_COMPLETE_REWARD = 100
LAP_FAIL_REWARD = -10


def parse_input():
    global old_progress, iters_not_moving, laps, gamepad
    # want to return 1d array with wheel slips, position, velocity, and normalized position.
    wheel_slips = list(info.physics.wheelSlip)
    velocity = list(info.physics.velocity)
    real_pos = list(info.graphics.carCoordinates)
    # 1d value tracking the distance along the spline. goes from 0-1
    progress = [info.graphics.normalizedCarPosition + 0]
    wheels_out = [info.physics.numberOfTyresOut]

    if round(velocity[0]) == 0:
        iters_not_moving += 1

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
    global old_progress, laps, iters_not_moving, gamepad
    gamepad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_RIGHT_SHOULDER)
    gamepad.update()
    time.sleep(0.1)
    gamepad.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_RIGHT_SHOULDER)
    gamepad.update()
    time.sleep(0.1)
    old_progress = info.graphics.normalizedCarPosition
    laps = info.graphics.completedLaps
    iters_not_moving = 0

def step(action):
    perform_output(action)
    vals = parse_input()
    delta = info.graphics.normalizedCarPosition - old_progress
    done = False
    reward = delta
    if info.graphics.completedLaps > laps:
        done = True
        reward += LAP_COMPLETE_REWARD

    if info.physics.numberOfTyresOut > 2:
        done = True
        reward += LAP_FAIL_REWARD

    if iters_not_moving > NOT_MOVING_THRESHOLD:
        done = True
        reward += LAP_FAIL_REWARD

    reward += info.physics.speedKmh * SPEED_MULT

    return vals, reward, done