import sounddevice
import speech_recognition as sr
import threading
from typing import Callable


class VoiceRecognition():
    def __init__(self,
                 exit_event: threading.Event,
                 timeout: float = 10,
                 phrase_time_limit: float = 10):
        self.subscribers: dict[str, Callable] = dict()
        self.exit_event = exit_event

        # the voice recognition thread should sleep (wait) until
        # the gui thread notify it that it needs voice recogintion
        self.need_recording_cond: threading.Condition = threading.Condition()
        self.need_recording = False

        self.timeout = timeout
        self.phrase_time_limit = phrase_time_limit

    def register_subscriber(self, subscriber_name: str, fn: Callable):
        self.subscribers[subscriber_name] = fn

    def notify_subscriber(self, subscriber_name: str, *args):
        self.subscribers[subscriber_name](*args)

    def voice_start_handler(self):
        """
        when the gui thread needs voice recognition, it will use
        this call back function to notify the voice recognition
        thread to wake up.

        This method will be called on the gui thread since it's a
        callback function.
        """
        self.need_recording = True
        with self.need_recording_cond:
            self.need_recording_cond.notify()

    def voice_exit_handler(self):
        """
        callback function for when the exit_event is set in another thread
        and this thread needs to wake up
        """
        with self.need_recording_cond:
            self.need_recording_cond.notify()

    def start_voice_input(self):
        """
        Records voice input from the microphone.
        """
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            while True:
                with self.need_recording_cond:
                    while not self.need_recording:
                        if self.exit_event.is_set():
                            break
                        print("waiting for gui thread to request voice rec")
                        self.need_recording_cond.wait()
                    if self.exit_event.is_set():
                        break
                    try:
                        print("started listening")
                        audio = recognizer.listen(
                            source,
                            timeout=self.timeout,
                            phrase_time_limit=self.phrase_time_limit
                        )
                        text = recognizer.recognize_google(audio)
                        self.need_recording = False
                        print(f"Voice input recognized: {text}")
                        self.notify_subscriber(
                            "llama-add-prompt",
                            "voice_rec",
                            text
                        )
                    except sr.UnknownValueError:
                        print("Sorry, I couldn't understand the audio.")
                    except sr.RequestError as e:
                        print(f"Error with the speech recognition service: {e}")
                        self.exit_event.set()
                        break
                    except Exception as e:
                        print(f"Error during voice recording: {e}")
                        self.exit_event.set()
                        break
