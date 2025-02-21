import socket
from typing import Callable
from events import Events, TCPServerEvents


# Red rover at dorm
IP = "10.49.90.168"
PORT = 5432


class TCPServer():
    def __init__(self, ip_address: str = IP, port: int = PORT):
        self.ip_address = ip_address
        self.port = port
        self.event_subscribers : dict[Events, dict[str, Callable]] = dict()
        
    
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

    def start_server(self) -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
            server.bind((self.ip_address, self.port))
            server.listen()
            while True:
                client, addr = server.accept()
                with client:
                    print(f"Connected by {addr}")
                    while True:
                        data: bytes = client.recv(1024) # 1024 should be enough for a few words
                        if not data:
                            break
                        print(data)
                        self.notify_event_all_subscribers(TCPServerEvents.MSG_RECEIVED, data)
    
                        
if __name__ == "__main__":
    t = TCPServer()
    t.start_server()
            
            
    
