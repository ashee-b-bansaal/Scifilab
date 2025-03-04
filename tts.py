import queue
import logging
from RealtimeTTS import TextToAudioStream, SystemEngine, AzureEngine, ElevenlabsEngine, GTTSEngine
import queue
from typing import Callable
import threading
import traceback
import time
from dotenv import load_dotenv
import os
from enum import Enum


FEMALE_VOICE="en-US-JennyNeural"
MALE_VOICE="en-US-GuyNeural"

load_dotenv()

AZURE_SPEECH_KEY = os.getenv('AZURE_SPEECH_KEY')

class Emotions(Enum):
    SAD="sad"
    HAPPY="cheerful"
    NEUTRAL="neutral"
    ANGRY="angry"
    TERRIFIED="terrified"

    # need angry, happy, sad, terrified, neutral


class TTS():
    def __init__(self, finished_speaking_handler: Callable, output_device_index=None,gender="female"):
        self.tts_q: queue.Queue = queue.Queue()
        self.voice= FEMALE_VOICE if gender=="female" else MALE_VOICE
        self.need_tts: bool = False
        self.need_tts_cond: threading.Condition = threading.Condition()
        self.tts_engine = AzureEngine(service_region="eastus",
                                      speech_key=AZURE_SPEECH_KEY,
                                      voice=self.voice
                                      )
        self.output_device_index=output_device_index
        self.tts_stream = TextToAudioStream(
            engine=self.tts_engine,
            on_audio_stream_stop = finished_speaking_handler,
            level=logging.INFO,
            frames_per_buffer=248,
            output_device_index=self.output_device_index
        )
        self.text_queue: queue.Queue = queue.Queue()

    def add_text_handler(self, text:str):
        self.text_queue.put_nowait(text)
    
    def add_emotion_handler(self, emotion:Emotions):
        text = self.text_queue.get_nowait()
        self.add_tts_handler(emotion, text)

    def add_tts_handler(self, emotion:Emotions, text:str):
        self.tts_q.put_nowait((emotion.value,text))
        with self.need_tts_cond:
            self.need_tts = True
            self.need_tts_cond.notify()

    def start_tts(self):
        try:
            while True:
                with self.need_tts_cond:
                    while not self.need_tts or self.tts_q.empty():
                        self.need_tts_cond.wait()
                    emotion,text = self.tts_q.get_nowait()
                    print("tts emotion and text",emotion, text)
                    self.tts_engine.set_emotion(emotion, emotion_degree=2.0)
                    self.tts_stream.feed(text)
                    self.tts_stream.play()
                    self.need_tts = False
        except:
            traceback.print_exc()

            
    
if __name__ == "__main__":
    a = TTS(lambda: print("done"), gender="female")
    tts_thread: threading.Thread = threading.Thread(target = a.start_tts, daemon=True)
    tts_thread.start()
    a.add_tts_handler(Emotions.SAD,"The dog is running")
    time.sleep(2)

    a.add_tts_handler(Emotions.ANGRY,"the weather is so nice today")
    time.sleep(5)
    print("done")

#     speech_config = speechsdk.SpeechConfig(subscription=os.environ.get('AZURE_SPEECH_KEY'), region="eastus")
#     audio_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)
#     speech_config.speech_synthesis_voice_name='en-US-AvaMultilingualNeural'
#     speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)
#     ssml = """
#     <speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xmlns:mstts="https://www.w3.org/2001/mstts" xml:lang="en-US">
#     <voice name="en-US-AvaMultilingualNeural">
#         <mstts:express-as style="cheerful" styledegree="2">
#             That'd be just amazing!
#         </mstts:express-as>
#         <mstts:express-as style="my-custom-style" styledegree="0.01">
#             What's next?
#         </mstts:express-as>
#     </voice>
# </speak>
#     """
#     speech_synthesizer.speak_ssml_async(ssml).get()



    

