import time
import copy
import threading 
from typing import Callable
import cv2

class OpenCVCamera():
    def __init__(self,
                 source_index: int,
                 cam_height: int,
                 cam_width: int,
                 cam_fps: float,
                 new_frame_handler: Callable = lambda x: x) -> None:
        self.source_index = source_index
        self.cam = cv2.VideoCapture(self.source_index)
        self.cam_height = cam_height
        self.cam_width = cam_width
        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_height)
        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, cam_width) 
        self.cam.set(cv2.CAP_PROP_FPS, cam_fps)
        self.cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.new_frame_handler = new_frame_handler


        self.need_response = False
        self.need_response_cond = threading.Condition()

    def read(self):
        return self.cam.read()

    # the 2 functions below are for multithreading purposes

    def need_new_frame(self):
        with self.need_response_cond:
            self.need_response = True
            self.need_response_cond.notify()

    def start_recording(self):
        while True:
            print("hello")
            with self.need_response_cond:
                while not self.need_response:
                    self.need_response_cond.wait()
                ret, frame = self.cam.read()
                if not ret:
                    print("cant parse frame")
                    break
                self.new_frame_handler(copy.deepcopy(frame))
                self.need_response = False



                
if __name__ == "__main__":
    def print_type(x):
        print(x.shape)
        cv2.imshow("hello", x)
        cv2.waitKey(1)
        cv2.waitKeyEx(1)

    a = OpenCVCamera(0, 720, 1280, 30.0 ,print_type)
    cam_thread = threading.Thread(target = a.start_recording, daemon = True)
    cam_thread.start()
    while True:
        a.need_new_frame()
    
    
                

            
