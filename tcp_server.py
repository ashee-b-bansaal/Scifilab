import time
import threading
import traceback
import queue
import logging
import socket
from typing import Callable
from events import Events, TCPServerEvents

# Red rover at dorm
IP = "10.49.89.239"
PORT = 5432

class TCPServer():
    def __init__(self, logger: logging.Logger, ip_address: str = IP, port: int = PORT):
        self.logger = logger
        self.ip_address = ip_address
        self.port = port
        self.event_subscribers : dict[Events, dict[str, Callable]] = dict()
        
        self.msg_q: queue.Queue = queue.Queue()
        self.new_msg_to_write = False
        self.new_msg_to_write_cond:threading.Condition = threading.Condition()

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

    def notify_event_all_subscribers(
            self,
            event: Events, *args):
        if event not in self.event_subscribers:
            return
        for d in self.event_subscribers[event].keys():
            self.notify_event_subscriber(event, d, *args)

    def new_msg_to_write_handler(self, msg: str):
        self.msg_q.put_nowait(msg)
        with self.new_msg_to_write_cond:
            self.new_msg_to_write = True
            self.new_msg_to_write_cond.notify()
            
    def write_to_client_socket(
            self,
            client_socket,
            addr):
        try:
           while True:
                with self.new_msg_to_write_cond:
                    self.new_msg_to_write_cond.wait_for(lambda: self.new_msg_to_write or self.msg_q.qsize() != 0)
                    text: str = self.msg_q.get_nowait()
                    print("text to write is", text)
                    text = text + "\n"
                    client_socket.send(text.encode('utf-8'))
                    self.logger.debug(f"TCPServer sending: text sent to {addr} is {text}")
                    self.new_msg_to_write = False
        except:
            self.logger.exception(f"writing to {addr} failed")

    def read_from_client_socket(
            self,
            client_socket,
            addr):
        try: 
            while True:
                data: bytes = client_socket.recv(1024)
                self.logger.debug(f"TCPServer reading: data from {addr} is : {data.decode('utf-8')}")
                print(data)
                if not data:
                    break
                self.notify_event_all_subscribers(TCPServerEvents.MSG_RECEIVED, data)
        except:
            self.logger.exception(f"client {addr} disconneted")

    def start_server(self) -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
            server.bind((self.ip_address, self.port))
            server.listen()
            while True:
                client, addr = server.accept()
                self.logger.debug(f"TCPServer: new connection {client} from {addr}")
                with client:
                    write_thread = threading.Thread(
                        target=self.write_to_client_socket,
                        args = (client, addr),
                        daemon=True)
                    write_thread.start()

                    read_thread = threading.Thread(
                        target=self.read_from_client_socket,
                        args = (client, addr),
                        daemon=True)
                    read_thread.start()
                    while write_thread.is_alive() and read_thread.is_alive():
                        time.sleep(0.1)

                        
if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    t = TCPServer(logger)
    thread_server = threading.Thread(target=t.start_server, daemon=True)
    thread_server.start()

    cnt = 0
    while True:
        t.new_msg_to_write_handler(f"hello{cnt}\n")
        time.sleep(1)
        cnt += 1
    
            
            
    
