import logging
import time
import queue
import threading
from typing import Callable
from events import *
from tcp_server import TCPServer


class AndroidInput():
    """
    this class checks if its message queue is empty or not
    if not then send to llm.
    message queue is filled using tcp server
    """
    def __init__(self, logger: logging.Logger) -> None:
        self.logger = logger
        self.event_subscribers : dict[Events, dict[str, Callable]] = dict()
        self.need_response_cond : threading.Condition = threading.Condition()
        self.need_response = False

        self.msg_q: queue.Queue = queue.Queue()

    def register_event_subscriber(self,
                                  event: Events,
                                  subscriber_name: str, fn: Callable):
        if event not in self.event_subscribers:
            self.event_subscribers[event] = dict()
        self.event_subscribers[event][subscriber_name] = fn

    def notify_event_subscriber(self,
                                event: Events,
                                subscriber_name: str, *args):
        if event in self.event_subscribers:
            self.event_subscribers[event][subscriber_name](*args)

    def notify_event_all_subscribers(self,
                                    event: Events, *args):
        if event not in self.event_subscribers:
            return
        for d in self.event_subscribers[event].keys():
            self.notify_event_subscriber(event, d, *args)
    
    def voice_input_ready_handler(self):
        with self.need_response_cond:
            self.need_response = True
            self.need_response_cond.notify()

    def server_msg_received_handler(self, msg: bytes):
        self.msg_q.put_nowait(msg.decode("utf-8"))
        with self.need_response_cond:
            self.need_response_cond.notify()
        
    def start_input(self) -> None:
        while True:
            with self.need_response_cond:
                while not self.need_response or self.msg_q.empty():
                    self.need_response_cond.wait()
                msg: str = self.msg_q.get_nowait()
                self.logger.info(f"the android phone sent this: {msg}")
                self.notify_event_subscriber(
                    AndroidInputEvents.INPUT_READY,
                    "llm",
                    msg
                )
            print("A")
                    

            
if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    t = TCPServer(logger)
    a = AndroidInput(logger)
    t.register_event_subscriber(
        TCPServerEvents.MSG_RECEIVED,
        "android_input",
        a.server_msg_received_handler
    )
    server_thread = threading.Thread(target=t.start_server, daemon=True)
    server_thread.start()

    while True:
        print(a.msg_q.qsize())
        time.sleep(2)
    

    



