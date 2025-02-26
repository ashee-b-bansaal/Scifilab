import os
import cv2
import threading
import queue
from typing import Callable, Tuple
import traceback


class VideoRecorder():
    """
    write frames in a queue into a file. might need to rewrite using multiprocessing
    instead of threading to improve perf.

    video_filename: [sth].mp4
    """
    def __init__(self,
                 video_folder_path: str,
                 video_filename: str,
                 frame_size: Tuple[int, int] = (1280, 720)):
        self.video_folder_path = video_folder_path
        self.video_filename = video_filename
        self.video_file_path = os.path.join(self.video_folder_path, self.video_filename)
        self.four_cc = cv2.VideoWriter_fourcc(*'mp4v')
        self.frame_size: Tuple[int, int] = frame_size
        self.FPS = 30.0
        self.video_writer = cv2.VideoWriter(
            self.video_file_path,
            self.four_cc,
            self.FPS,
            self.frame_size)
        
        self.frames_q: queue.Queue = queue.Queue()
        self.need_write_new_frame_cond: threading.Condition = threading.Condition()
        self.need_write_new_frame = False

        self.event_subscribers: dict[str, dict[str, Callable]] = dict()
        self.need_exit = False

    def register_event_subscriber(self,
                                  event_name: str,
                                  subscriber_name: str, fn: Callable):
        if event_name not in self.event_subscribers:
            self.event_subscribers[event_name] = dict()
        self.event_subscribers[event_name][subscriber_name] = fn

    def notify_event_subscriber(self,
                                event_name: str,
                                subsciber_name: str, *args):
        self.event_subscribers[event_name][subsciber_name](*args)

    def new_frame_event_handler(self, new_frame):
        self.frames_q.put_nowait(new_frame)
        self.need_write_new_frame = True
        with self.need_write_new_frame_cond:
            self.need_write_new_frame_cond.notify()

    def exit_event_handler(self):
        self.frames_q.put_nowait(None)
        with self.need_write_new_frame_cond:
            self.need_exit = True
            self.need_write_new_frame_cond.notify()
        print("video exit handler finished")

    def graceful_exit(self):
        self.video_writer.release()
        print("video released")

    def write_video(self):
        try:
            while True:
                with self.need_write_new_frame_cond:
                    while not self.need_write_new_frame or not self.need_exit:
                        if self.need_exit:
                            self.graceful_exit()
                            break
                        self.need_write_new_frame_cond.wait()
                if self.need_exit:
                    self.graceful_exit()
                    break
                frame = self.frames_q.get_nowait()
                if frame is None:
                    self.graceful_exit()
                    break
                self.video_writer.write(frame)
                self.need_write_new_frame = False
        except:
            traceback.print_exc()
        print("done")


if __name__ == "__main__":
    pass
    
