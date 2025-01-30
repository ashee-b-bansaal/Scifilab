import threading
from typing import Callable
import queue
from ollama import chat

# prompt = "Generate an 1 sentence sad response, neutral response, happy response to the" \
#     "following using context from our conversation: "
prompt = ""

class Llama():
    def __init__(self, exit_event: threading.Event):
        self.exit_event = exit_event
        self.subscribers: dict[str, Callable] = dict()

        # the voice recognition thread will put the text
        # into this queue via the add_prompt_handler
        self.prompt_q: queue.Queue = queue.Queue()

        # llm thread should sleep (wait) until the voice rec
        # thread notify it that it needs a response
        self.need_response_cond: threading.Condition = threading.Condition()
        self.need_response: bool = False

        # message history for ollama
        self.messages = []

    def register_subscriber(self, subscriber_name: str, fn: Callable):
        self.subscribers[subscriber_name] = fn

    def notify_subscriber(self, subsciber_name: str, *args):
        self.subscribers[subsciber_name](*args)

    def llama_exit_handler(self):
        with self.need_response_cond:
            self.need_response_cond.notify()

    def add_prompt_handler(self, new_prompt: str):
        self.prompt_q.put_nowait(prompt + new_prompt)
        self.need_response = True
        with self.need_response_cond:
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
                    new_prompt: str = self.prompt_q.get_nowait()

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

                    print("llm response is", response_content)
                    self.need_response = False
                except Exception as e:
                    print(e)
                    self.exit_event.set()
                    break
