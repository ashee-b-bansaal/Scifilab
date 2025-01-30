import cv2
import numpy as np
import threading
from typing import Callable
from voice_recognition import VoiceRecognition
from llama_module import Llama, NUMBER_OF_OPTIONS
from functools import partial


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

        # more robust is a State Machine to transition between states
        # right now, the render_order and render_ready are used to determine
        # what the current state is

        # variables for llm rendering
        self.llm_options: list[str] = []
        self.selected_llm_option_index = 0
        self.number_of_llm_options = NUMBER_OF_OPTIONS

    def register_subscriber(self, subscriber_name: str, fn: Callable):
        self.subscribers[subscriber_name] = fn

    def notify_subscriber(self, subsciber_name: str, *args):
        self.subscribers[subsciber_name](*args) 

    def add_ui_component(self,
                         component_name: str,
                         component_render_fn: Callable):
        self.render_order.append(component_name)
        self.render_functions[component_name] = component_render_fn
        self.render_ready[component_name] = False

    def update_canvas(self):
        """
        draws everything on canvas
        """
        for ui_element in self.render_order:
            if self.render_ready[ui_element]:
                self.render_functions[ui_element]()

    def handle_input(self):
        pressed_key = cv2.waitKeyEx(1) & 0xFF
        if pressed_key != 255:
            print(pressed_key)
        
        if pressed_key == 27:
            self.exit_event.set()

            # there is probably a better way to do this
            # this is letting the subsciber handle the exit
            for subscriber in list(self.subscribers.keys()):
                if "-exit" in subscriber:
                    self.notify_subscriber(subscriber)
        elif pressed_key == ord(' '):
            self.notify_subscriber("voice-start")
        elif pressed_key == 84:  # up arrow:
            if self.render_ready['llm-options']:
                self.selected_llm_option_index = (self.selected_llm_option_index + 1) % self.number_of_llm_options
        elif pressed_key == 82:  # down arrow
            if self.render_ready['llm-options']:
                self.selected_llm_option_index = (self.selected_llm_option_index - 1) % self.number_of_llm_options
        elif pressed_key == 13:  # enter key
            if self.render_ready['llm-options']:
                self.notify_subscriber(
                    "llama-add-prompt",
                    "ui",
                    self.llm_options[self.selected_llm_option_index]
                )
                self.render_ready["llm-options"] = False

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

    def render_llm_options(self):
        for i in range(len(self.llm_options)):
            # print(self.llm_options[i])
            thickness = 1 
            if i == self.selected_llm_option_index:
                thickness = 2
            cv2.putText(
                self.canvas,
                self.llm_options[i],
                (0, 50 + 30 * i),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                thickness, cv2.LINE_AA)

    def llama_response_handler(self, response: str):
        # ui_component_name = "llm-options"
        # responses_list = response.splitlines()
        # responses_list = [rep for rep in responses_list if rep[0].isdigit()]
        # def render_options():
        # self.render_functions[ui_component_name]
        print("response recieved by gui thread")
        self.llm_options = [rep for rep in response.splitlines() if len(rep) > 2 and rep[0].isdigit()]
        print("responses are: ", self.llm_options)
        self.render_functions["llm-options"] = lambda: self.render_llm_options()
        self.render_ready["llm-options"] = True


if __name__ == "__main__":
    exit_event: threading.Event = threading.Event()
    gui: GUIClass = GUIClass(exit_event)
    voice_rec: VoiceRecognition = VoiceRecognition(exit_event)
    llama: Llama = Llama(exit_event)

    gui.register_subscriber("voice-exit", voice_rec.voice_exit_handler)
    gui.register_subscriber("voice-start", voice_rec.voice_start_handler)
    gui.register_subscriber("llama-exit", llama.llama_exit_handler)
    gui.register_subscriber("llama-add-prompt", llama.add_prompt_handler)

    voice_rec.register_subscriber("llama-add-prompt", llama.add_prompt_handler)

    llama.register_subscriber(
        "gui-recieve-llama-response",
        gui.llama_response_handler
    )

    gui.add_ui_component("llm-options", lambda x: x,)

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
