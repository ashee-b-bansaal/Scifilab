import logging
import base64
import time
import traceback
import threading
from typing import Callable
import os
from dotenv import load_dotenv
from openai import OpenAI
import queue
import cv2
load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

NUMBER_OF_OPTIONS = 3 #this should be 4

default_img = cv2.imread("istockphoto-1315588490-612x612.jpg")

DEFAULT_CONTEXT =  "Your role is to help facilitate a conversation happening between person A and person B. You will be given what person A says. You have to generate 3 short 1-sentence responses each on its own line with a number at the start for person B based on a few keywords. Person B will then choose one of these 3 responses and you should remember their response."

def generate_person_a_prompt(voice_rec_input: str, kbd_input: str):
    return f"Person A says: \"{voice_rec_input}\"." \
        f"Keywords for Person B are: {kbd_input}. "

def encode_image(image):
    retval, buffer = cv2.imencode('.jpg', image)
    encoded_img = base64.b64encode(buffer).decode("utf-8")
    return encoded_img
    
class ChatGPTAPI():
    def __init__(self, logger: logging.Logger, model: str = "gpt-4o-mini", use_surrounding_context: bool = False):
        self.use_surrounding_context = use_surrounding_context
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
        
        self.messages: list[dict] = []
        self.client = OpenAI()
        
        self.initialized_context = False
        self.initialized_context_cond: threading.Condition = threading.Condition()

        if not self.use_surrounding_context:
            self.initialize_chat()
            

    def initialize_chat(self, msg: str = DEFAULT_CONTEXT):
        with self.initialized_context_cond: 
            self.messages = [{
                "role": "developer",
                "content": msg
            }]
            self.initialized_context = True
            self.initialized_context_cond.notify()

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

    
    def new_context_image_handler(self,
                                  frame = default_img):
        """
        the interaction camera thread will call this method, which sends the
        context image to chat gpt which sends back a string describing the
        context of the situation to be used to initilize the chat.
        """
        image_client = OpenAI()
        response = image_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Person A is the one in the picture, person B is the one taking the picture. Can you describing the relationship between person A and person B and make it general, not too specific and short in 1 sentence ",
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{encode_image(frame)}"},
                        },
                    ],
                }
            ],
        )
        additional_context = response.choices[0].message.content
        print(additional_context)
        self.initialize_chat(additional_context + DEFAULT_CONTEXT)
       

    def keyword_input_handler(self, kbd_input: str):
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
            # wait until the context is initialized
            if self.use_surrounding_context:
                with self.initialized_context_cond: 
                    while not self.initialized_context:
                        self.initialized_context_cond.wait()
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
    img = cv2.imread('istockphoto-1315588490-612x612.jpg')
    gpt.new_context_image_handler(img)
    # llm_thread = threading.Thread(target=gpt.start_conversation, daemon=True)
    # llm_thread.start()
    # print("seting up, please wait")
    # time.sleep(5)

    # turn_cnt = 0

    # while True:
    #     # Person A's turn
    #     if turn_cnt % 2 == 0:
    #         a_text = input("Please enter what person A says:")
    #         b_keywords = input(
    #             "Please enter the keywords to generate response for person B")
    #         if a_text == "exit" or b_keywords == "exit":
    #             break
    #         gpt.add_prompt_handler(
    #             "A", generate_person_a_prompt(a_text, b_keywords))
    #         turn_cnt += 1
    #     else:
    #         b_choice = input("Please choose a response: ")
    #         if b_choice == "exit":
    #             break
    #         gpt.add_prompt_handler(
    #             "B", b_choice)
    #         turn_cnt += 1
