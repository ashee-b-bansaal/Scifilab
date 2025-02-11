import threading
from typing import Callable
import queue
from ollama import chat

NUMBER_OF_OPTIONS = 3


def generate_user_prompt(voice_rec_input: str, kbd_input: str):
    return f"This is what person A said: \"{voice_rec_input}\"." \
        f"Can you generate 3 very short 1 sentence responses that is based on these keywords: {kbd_input}. " \
        "Each response should be on numbered and on its own line"\
        "(example: 1., 2. , 3.)"


ui_prompt = "I choose this response, don't response, just remember what I said:"


class Llama():
    def __init__(self, exit_event: threading.Event):
        self.exit_event = exit_event
        self.subscribers: dict[str, Callable] = dict()

        # the voice recognition thread will put the text
        # into this queue via the add_prompt_handler
        self.voice_rec_q: queue.Queue = queue.Queue()

        self.prompt_q: queue.Queue = queue.Queue()

        # llm thread should sleep (wait) until the voice rec
        # thread notify it that it needs a response
        self.need_response_cond: threading.Condition = threading.Condition()
        self.need_response: bool = False

        # message history for ollama
        self.messages: list[dict] = []

    def register_subscriber(self, subscriber_name: str, fn: Callable):
        self.subscribers[subscriber_name] = fn

    def notify_subscriber(self, subsciber_name: str, *args):
        self.subscribers[subsciber_name](*args)

    def llama_exit_handler(self):
        with self.need_response_cond:
            self.need_response_cond.notify()

    def keyboard_input_handler(self, kbd_input: str):
        """
        the event is "keyboard_input", the keyboard thread calls this method
        once the user has inputted something. The assumption is that the
        voice rec thread has already added the user voice to self.voice_rec_q
        before this method is called, so after the keyboard input, the llama
        thread has all it needs to generate a response so it needs to wake up
        """
        voice_rec_input = self.voice_rec_q.get_nowait()
        self.add_prompt_handler(
            "user", generate_user_prompt(voice_rec_input, kbd_input))

    def voice_rec_input_handler(self, voice_rec_input: str):
        print("voice_rec_input is", voice_rec_input)
        self.voice_rec_q.put_nowait(voice_rec_input)

    def add_prompt_handler(self, sender: str, new_prompt: str):
        self.prompt_q.put_nowait(
            (sender,
             (ui_prompt + new_prompt if sender == "ui" else new_prompt)))
        with self.need_response_cond:
            self.need_response = True
            self.need_response_cond.notify()

    def start_conversation(self):
        while True:
            with self.need_response_cond:
                try:
                    while not self.need_response:
                        if self.exit_event.is_set():
                            break
                        print("waiting for voice recognition to send")
                        self.need_response_cond.wait()
                    if self.exit_event.is_set():
                        break
                    sender, new_prompt = self.prompt_q.get_nowait()
                    # has to vary behavior based on sender maybe?
                    # sender can be gui thread which chooses a response
                    # or the voice rec thread which gives the llm the user's
                    # voice input

                    # if gui is sending then response will be given to tts
                    # if voice_rec is sending then response will be given
                    # to gui to display
                    print("prompt is", new_prompt)
                    new_message = {
                        'role': 'user',
                        'content': new_prompt,
                    }
                    response = chat(
                        model='llama3.2',
                        messages=self.messages + [new_message],
                    )
                    response_content = response['message']['content']
                    self.messages += [
                        new_message,
                        {'role': 'user', 'content': response_content}
                    ]


                    self.need_response = False

                    if sender == "ui":
                        # do tts, have to synchronize threads
                        # if tts is done then this will call the registered voice rec call back
                        # to request user voice input
                        pass
                    elif sender == "user":
                        self.notify_subscriber(
                            "gui-recieve-llama-response",
                            response_content
                        )
                    print(f"llm response for {sender} is", response_content)
                except Exception as e:
                    print(e)
                    self.exit_event.set()
                    break
