import pyttsx3
import queue
from typing import Callable
import threading
import traceback
import time


class TTS():
    def __init__(self, finished_speaking_handler: Callable):
        self.engine = pyttsx3.init()
        self.tts_q: queue.Queue = queue.Queue()

        self.engine.connect('finished-utterance', finished_speaking_handler)

        self.need_tts: bool = False
        self.need_tts_cond: threading.Condition = threading.Condition()
        self.finished_speaking_handler = finished_speaking_handler

    def add_tts_handler(self, text):
        self.tts_q.put_nowait(text)
        with self.need_tts_cond:
            self.need_tts = True
            self.need_tts_cond.notify()

    def exit_handler(self):
        self.tts_q.put_nowait(None)
        with self.need_tts_cond:
            self.need_tts = True
            self.need_tts_cond.notify()

    def start_tts(self):
        try:
            with self.need_tts_cond:
                while True:
                    while not self.need_tts:
                        self.need_tts_cond.wait()
                    text = self.tts_q.get_nowait()
                    if text is None:
                        break
                    print(text)
                    self.engine.say(text, 'text')
                    self.engine.runAndWait()
                    self.need_tts = False
                    self.finished_speaking_handler()
                    print("done speak")
            print("tts thread exited")
        except:
            traceback.print_exc()

def print_bruh(name: str, comp: bool):
    print("hello mate")

if __name__ == "__main__":
    a = TTS(print_bruh)
    tts_thread: threading.Thread = threading.Thread(target = a.start_tts)
    tts_thread.start()
    time.sleep(1)
    a.add_tts_handler("brother my hand is cold end")
    time.sleep(2)
    a.add_tts_handler("the weather is so nice today end")
    time.sleep(4)
    a.add_tts_handler(None)
    tts_thread.join()
    print("done")
