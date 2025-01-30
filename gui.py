import cv2
import numpy as np
import threading
from typing import Callable
from voice_recognition import VoiceRecognition
from llama_module import Llama

class GUIClass():
    """
    The GUI class is responsible for displaying the GUI.

    As this is simulating an AR app, all gui objects will be drawn
    onto the frame recieved by the camera.

    """
    def __init__(self, exit_event: threading.Event):

        # list of the subscribers (str) along with their methods.
        # when an event of interest occurs, the gui object will
        # call the corresponding method of the subscriber
        self.subscribers: dict[str, Callable] = dict()

        # the order by which to render the elements.
        self.render_order: list[str] = []

        # key is name : str, value is function to draw on canvas
        self.render_functions: dict[str, Callable] = dict()

        # whether the thing to render is ready to be rendered
        # (call the function in render_functions).
        # key is str, value is bool
        self.render_ready: dict[str, bool] = dict()

        # canvas to draw everything on, in the render method,
        # this should be updated as the camera read frames.
        self.canvas = np.zeros(0)

        self.frame_height = -1
        self.frame_width = -1

        # if exit_event is set then must end.
        self.exit_event = exit_event

    def register_subscriber(self, subscriber_name: str, fn: Callable):
        self.subscribers[subscriber_name] = fn

    def notify_subscriber(self, subsciber_name: str, *args):
        self.subscribers[subsciber_name](*args)

    def update_canvas(self):
        """
        draws everything on canvas
        """
        for ui_element in self.render_order:
            if self.render_ready[ui_element]:
                self.render_functions[ui_element]()

    def handle_input(self):
        pressed_key = cv2.waitKeyEx(1) & 0xFF
        if pressed_key == 27:
            self.exit_event.set()

            # there is probably a better way to do this
            # this is letting the subsciber handle the exit
            for subscriber in list(self.subscribers.keys()):
                if "-exit" in subscriber:
                    self.notify_subscriber(subscriber)
        elif pressed_key == ord(' '):
            self.notify_subscriber("voice-start")

    def render(self):
        cam = cv2.VideoCapture(0)
        self.frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

        while True:
            ret, self.canvas = cam.read()

            if not ret:
                print("can't parse frame")
                break

            self.update_canvas()
            self.handle_input()

            cv2.imshow("realtime llm", self.canvas)

            if self.exit_event.is_set():
                break


if __name__ == "__main__":
    exit_event = threading.Event()
    gui = GUIClass(exit_event)
    voice_rec = VoiceRecognition(exit_event)
    llama = Llama(exit_event)

    gui.register_subscriber("voice-exit", voice_rec.voice_exit_handler)
    gui.register_subscriber("voice-start", voice_rec.voice_start_handler)
    gui.register_subscriber("llama-exit", llama.llama_exit_handler)

    voice_rec.register_subscriber("llama-add-prompt", llama.add_prompt_handler)
    
    voice_rec_thread: threading.Thread = threading.Thread(
        target=voice_rec.start_voice_input
    )
    llama_thread: threading.Thread = threading.Thread(
        target=llama.start_conversation
    )
    
    voice_rec_thread.start()
    llama_thread.start()
    gui.render()

    voice_rec_thread.join()
    llama_thread.join()
