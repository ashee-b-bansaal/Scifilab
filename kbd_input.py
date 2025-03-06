import logging
from typing import Callable
import threading
import queue


class KeyboardInput():
    def __init__(self, logger: logging.Logger) -> None:
        self.logger = logger
        self.event_subscribers : dict[str, dict[str, Callable]] = dict()
        self.need_response_cond : threading.Condition = threading.Condition()
        self.need_response = False

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

    def voice_input_ready_handler(self):
        with self.need_response_cond:
            self.need_response = True
            self.need_response_cond.notify()

    def start_input(self):
        with self.need_response_cond:
            print("hello")
            while True:
                while not self.need_response:
                    self.need_response_cond.wait()
                kbd_input = input("Please enter the keywords to generate the responses to what the hearing person just spoke: ")
                self.notify_event_subscriber(
                    "keyboard_input_ready",
                    "llm",
                    kbd_input
                )
                self.need_response = False

