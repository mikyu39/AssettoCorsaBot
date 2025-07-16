from pyvjoystick import vjoy

# Pythonic API, item-at-a-time
j = vjoy.VJoyDevice(1)

# turn button number 15 on
j.set_button(15, 1)

# Notice the args are (buttonID,state) whereas vJoy's native API is the other way around.


# turn button 15 off again
j.set_button(15, 0)

# Set X axis to fully left
j.set_axis(vjoy.HID_USAGE.X, 0x1)

# Set X axis to fully right
j.set_axis(vjoy.HID_USAGE.X, 0x8000)

# Also implemented:

j.reset()
j.reset_buttons()
j.reset_povs()


# The 'efficient' method as described in vJoy's docs - set multiple values at once

print(j._data)
# >> > <pyvjoystick.vjoy._sdk._JOYSTICK_POSITION_V2 at 0x.... >


j._data.lButtons = 19  # buttons number 1,2 and 5 (1+2+16)
j._data.wAxisX = 0x1
j._data.wAxisY = 0x8000

# send data to vJoy device
j.update()
