import time
import traceback
import threading
from typing import Callable
import os
from dotenv import load_dotenv
from openai import OpenAI
import queue

load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

NUMBER_OF_OPTIONS = 3

def generate_person_a_prompt(voice_rec_input: str, kbd_input: str):
    return f"Person A says: \"{voice_rec_input}\"." \
        f"Keywords for Person B are: {kbd_input}. "


class ChatGPTAPI():
    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        self.event_subscribers: dict[str, dict[str, Callable]] = dict()
        self.need_response_cond: threading.Condition = threading.Condition()
        self.need_response = False

        # the voice recognition thread will put the text
        # into this queue via the add_prompt_handler
        self.voice_rec_q: queue.Queue = queue.Queue()

        # after the keyboard input is entered, a tuple of keyboard input and
        # text from voice_rec_q is added here
        self.prompt_q: queue.Queue = queue.Queue()

        self.messages: list[dict] = [{
            "role": "developer",
            "content": "Your role is to help facilitate a conversation happening between person A and person B. You will be given what person A says. You have to generate 3 short 1-sentence responses each on its own line with a number at the start for person B based on a few keywords. Person B will then choose one of these 3 responses and you should remember their response."
        }]
        self.client = OpenAI()

    def register_event_subscriber(self,
                                  event_name: str,
                                  subscriber_name: str, fn: Callable):
        if event_name not in self.event_subscribers:
            self.event_subscribers[event_name] = dict()
        self.event_subscribers[event_name][subscriber_name] = fn

    def notify_event_subscriber(self,
                                event_name: str,
                                subsciber_name: str, *args):
        self.event_subscribers[event_name][subsciber_name](*args)

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
            "A", generate_person_a_prompt(voice_rec_input, kbd_input))

    def voice_rec_input_handler(self, voice_rec_input: str):
        print("voice_rec_input is", voice_rec_input)
        self.voice_rec_q.put_nowait(voice_rec_input)

    def add_prompt_handler(self, sender: str, text: str):
        """
        There are 2 senders, A and B. A is the hearing person, B is the deaf person.
        """
        self.prompt_q.put_nowait(
            (sender, text))
        with self.need_response_cond:
            self.need_response = True
            self.need_response_cond.notify()

    def start_conversation(self):
        try:
            while True:
                with self.need_response_cond:
                    while not self.need_response:
                        self.need_response_cond.wait()
                    sender, text = self.prompt_q.get_nowait()
                    assert sender == "A" or sender == "B"
                    new_message = {
                        'role': 'user',
                        'content': text
                    }
                    self.messages.append(new_message)
                    response_content = self.client.chat.completions.create(
                        model=self.model,
                        messages=self.messages
                    ).choices[0].message.content
                    self.messages.append({
                        "role": "assistant",
                        "content": response_content
                    })
                    self.need_response = False
                    if sender == "A":
                        if "new-response" in self.event_subscribers:
                            self.notify_event_subscriber(
                                "new-response",
                                "gui",
                                response_content
                            )
        except:
            traceback.print_exc()


if __name__ == "__main__":
    gpt = ChatGPTAPI()
    llm_thread = threading.Thread(target=gpt.start_conversation, daemon=True)
    llm_thread.start()
    print("seting up, please wait")
    time.sleep(5)

    turn_cnt = 0

    while True:
        # Person A's turn
        if turn_cnt % 2 == 0:
            a_text = input("Please enter what person A says:")
            b_keywords = input(
                "Please enter the keywords to generate response for person B")
            if a_text == "exit" or b_keywords == "exit":
                break
            gpt.add_prompt_handler(
                "A", generate_person_a_prompt(a_text, b_keywords))
            turn_cnt += 1
        else:
            b_choice = input("Please choose a response: ")
            if b_choice == "exit":
                break
            gpt.add_prompt_handler(
                "B", b_choice)
            turn_cnt += 1
