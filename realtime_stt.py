from RealtimeSTT import AudioToTextRecorder
import threading
from typing import Callable
import traceback


class VoiceRecognition():
    def __init__(self,
                 exit_event: threading.Event,
                 recorder):
        self.event_subscribers: dict[str, dict[str, Callable]] = dict()
        self.exit_event = exit_event

        # the voice recognition thread should sleep (wait) until
        # the gui thread notify it that it needs voice recogintion
        self.need_recording_cond: threading.Condition = threading.Condition()
        self.need_recording = False

        self.recorder = recorder

    def register_event_subscriber(self,
                                  event_name: str,
                                  subscriber_name: str, fn: Callable):
        if event_name not in self.event_subscribers:
            self.event_subscribers[event_name] = dict()
        self.event_subscribers[event_name][subscriber_name] = fn

    def notify_event_subscriber(self,
                                event_name,
                                subscriber_name, *args):
        self.event_subscribers[event_name][subscriber_name](*args)

    def voice_start_handler(self):
        """
        when the gui thread needs voice recognition, it will use
        this call back function to notify the voice recognition
        thread to wake up.

        This method will be called on the gui thread since it's a
        callback function.
        """
        with self.need_recording_cond:
            self.need_recording = True
            self.need_recording_cond.notify()

    def voice_exit_handler(self):
        """
        callback function for when the exit_event is set in another thread
        and this thread needs to wake up
        """
        with self.need_recording_cond:
            self.need_recording_cond.notify()

    def on_transcription_finished(self, text):
        print("transcription is:",text)
        self.notify_event_subscriber(
            "voice_input_ready",
            "llama",
            str(text)
        )
        self.notify_event_subscriber(
            "voice_input_ready",
            "kbd_input"
        )
        print("____________")
        print(self.event_subscribers["voice_input_ready"]["gui"])
        print("__________")
        self.notify_event_subscriber(
            "voice_input_ready",
            "gui",
            str(text))

    def start_voice_input(self):
        """
        Records voice input from the microphone
        """
        print("hello")
        try:
            with self.need_recording_cond:
                while True:
                    if self.exit_event.is_set():
                        break
                    while not self.need_recording:
                        if self.exit_event.is_set():
                            break
                        self.need_recording_cond.wait()
                    print("wassup")
                    if self.exit_event.is_set():
                        break
                    self.recorder.text(self.on_transcription_finished)
        except:
            traceback.print_exc()


def process_text(text):
    print(type(text))
    print(text)


def print_start():
    print("hello, recording started")


def print_end():
    print("hello, recording ended")


def on_trans_fin(text):
    print(text + "wtf")



