from sim_info import SimInfo

info = SimInfo()

def parse_input():
    # want to return 1d array with wheel slips, position, velocity, and normalized position.
    wheel_slips = list(info.physics.wheelSlips)
    velocity = list(info.physics.velocity)
    real_pos = list(info.graphics.carCoordinates)
    progress = list(info.graphics.normalizedCarPosition)

    out = wheel_slips + velocity + real_pos + progress
    return out

def parse_output(gas, brake, steering_angle):
    info.physics.gas = gas
    info.physics.brake = brake
    info.physics.steeringAngle = steering_angle
