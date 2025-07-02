import time 
from time import sleep
import serial

ser = serial.Serial('COM3', 115200, write_timeout=1.0)
print("Arduino initialized")
    
def send_trigger_bytes():
#   byte_data = bytes([value])
   ser.write('ff'.encode())
   time.sleep(0.005)
   ser.write('00'.encode())
print("function defined")


if __name__ == "__main__":
    send_trigger_bytes()
    print("trigger sent")

