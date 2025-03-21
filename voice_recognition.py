import logging
import speech_recognition as sr
import threading
from typing import Callable
import traceback


class VoiceRecognition():
    def __init__(self,
                 logger: logging.Logger,
                 input_device_index: int = -1) -> None:
        self.event_subscribers: dict[str, dict[str, Callable]] = dict()
        self.logger = logger
        # the voice recognition thread should sleep (wait) until
        # the gui thread notify it that it needs voice recogintion
        self.need_recording_cond: threading.Condition = threading.Condition()
        self.need_recording = False
        self.input_device_index=input_device_index

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
        self.need_recording = True
        with self.need_recording_cond:
            self.need_recording_cond.notify()

    def start_voice_input(self):
        """
        Records voice input from the microphone.
        """
        recognizer = sr.Recognizer()
        # with sr.Microphone(device_index=self.input_device_index) as source: 
        #     print("adjusting for ambiant noise")
        #     recognizer.adjust_for_ambient_noise(source)
        with self.need_recording_cond:
            while True:
                while not self.need_recording:
                    print("waiting for gui thread to request voice rec")
                    self.need_recording_cond.wait()
                try:
                    print("started listening")
                    with sr.Microphone(device_index=self.input_device_index) as source:
                        audio = recognizer.listen(
                            source
                        )
                        text = recognizer.recognize_openai(audio, language="en")
                    self.need_recording = False
                    print(f"Voice input recognized: {text}")
                    print(self.event_subscribers["voice_input_ready"].keys())
                    self.notify_event_subscriber(
                        "voice_input_ready",
                        "llm",
                        str(text)
                    )
                    self.notify_event_subscriber(
                        "voice_input_ready",
                        "keyword_input"
                    )
                    self.notify_event_subscriber(
                        "voice_input_ready",
                        "gui",
                        str(text))
                    try: 
                        self.notify_event_subscriber(
                            "voice_input_ready",
                            "tcp"
                        )
                    except:
                        pass
                    print("done listening")
                except sr.WaitTimeoutError:
                    pass
                except sr.UnknownValueError:
                    print("Sorry, I couldn't understand the audio.")
                except sr.RequestError as e:
                    print(f"Error with the speech recognition service: {e}")
                    traceback.print_exc()
                    break
                except Exception as e:
                    print(f"Error during voice recording: {e}")
                    traceback.print_exc()
                    break
