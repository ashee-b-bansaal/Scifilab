from enum import Enum, auto


class Events():
    def make_event():
        return Events()

class TCPServerEvents(Events):
    SERVER_START = Events.make_event()
    SERVER_END = Events.make_event()
    CLIENT_CONNECTED = Events.make_event()
    MSG_RECEIVED = Events.make_event()
    
class AndroidInputEvents(Events):
    INPUT_READY = Events.make_event()
