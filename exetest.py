import subprocess
import time

# creating a forever loop
while 1 :
    subprocess.call("TASKKILL /F /IM acs.exe", shell=True)
    time.sleep(10)