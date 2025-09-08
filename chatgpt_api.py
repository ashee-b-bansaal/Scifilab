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

NUMBER_OF_OPTIONS = 3

default_img = cv2.imread("istockphoto-1315588490-612x612.jpg")

NUM_WORDS = 7

# DEFAULT_CONTEXT = f"Your role is to help facilitate a conversation happening between person A and person B. You will be given what person A says. You have to generate {NUMBER_OF_OPTIONS} short 1-sentence responses with less than {NUM_WORDS} words each on its own line with a number at the start for person B based on a few keywords. The responses should have different meaning but based on the same keywords. Person B will then choose one of these {NUMBER_OF_OPTIONS} responses and you should remember their response."

# DEFAULT_CONTEXT = f"Your role is to help facilitate a conversation happening between person A and person B. You will be given what person A says. You have to generate {NUMBER_OF_OPTIONS} short 1-sentence responses each on its own line with a number at the start for person B based on a few keywords. The length of the responses should not be greater than 10 words. and proportional to the length of input provided, if only 1-3 words are provided as input, the options should contain around 5-7 words and if 5-7 words are provided, then each option should contain 10 words. In any situation, the length of each option should not go beyond 10 words at all. All the keywords provided as input should be present in each option. Do not cut off any part of the input words provided, if the user inputs 7 words, make sure to add those 7 words in all of the options.  The responses should have different meaning but based on the same keywords. Person B will then choose one of these {NUMBER_OF_OPTIONS} responses and you should remember their response."
DEFAULT_CONTEXT = f"Your role is to help facilitate a conversation happening between person A and person B. You will be given what person A says. You have to generate {NUMBER_OF_OPTIONS} short 1-sentence responses each on its own line with a number at the start for person B strictly based on a few keywords provided for the query. The responses should also not add extraneous detail and only be based on the keywords provided for that query. The responses should also not exclude any detail from the keywords either. If only 1 or 2 keywords are provided then make sure the options you generate are under 7 words.If no keywords are provided then generate the most likely responses to what person A said under 7 words. Person B will then choose one of these {NUMBER_OF_OPTIONS} responses and you should remember their response as context to generate further responses. Person B could also say 'Pardon?' do signify to person A to repeat what they said."


def generate_person_b_prompt(voice_rec_input: str, kbd_input: str, image_context:str = ""):
    return f"Optional context: {image_context}. Person A says: \"{voice_rec_input}\"." \
        f"Please generate {NUMBER_OF_OPTIONS} short responses, max 15 words, for Person B based on the provided keywords. If only 1 or 2 keywords are provided then make sure the options you generate are under 7 words. Each response should appear on its own line. The keywords are:{kbd_input}. Make sure the options don't deviate"

def generate_person_b_prompt_no_keywords(voice_rec_input: str, image_context: str = ""):
    return f"Optional context: {image_context}. Person A says: \"{voice_rec_input}\". Please generate {NUMBER_OF_OPTIONS} repsonses each with different meaning under 7 words person B and on its own line based on what person A says. "

def encode_image(image):
    retval, buffer = cv2.imencode('.jpg', image)
    encoded_img = base64.b64encode(buffer).decode("utf-8")
    return encoded_img
    
class ChatGPTAPI():
    def __init__(self,
                 logger: logging.Logger,
                 model: str = "gpt-4o",
                 use_surrounding_context: bool = False,
                 context_filename: str = ""):
        self.logger = logger
        self.use_surrounding_context = use_surrounding_context
        self.model = model
        self.event_subscribers: dict[str, dict[str, Callable]] = dict()
        self.need_response_cond: threading.Condition = threading.Condition()
        self.need_response = False

        self.last_voice_rec_text = ""
        self.last_voice_rect_text_lock = threading.Lock()


        
        # after the keyboard input is entered, a tuple of keyboard input and
        # text from voice_rec_q is added here
        self.prompt_q: queue.Queue = queue.Queue()
        
        self.messages: list[dict] = []
        self.client = OpenAI()
        
        self.initialized_context = False
        self.initialized_context_cond: threading.Condition = threading.Condition()

        print(DEFAULT_CONTEXT)
        self.context_filename = context_filename
        if not self.use_surrounding_context:
            context = DEFAULT_CONTEXT
            self.logger.info(f"Context: {context}")
            self.initialize_chat(context)
            print("THE CONTEXT IS", context)

        self.has_new_image_cond: threading.Condition = threading.Condition()
        self.has_new_image = False
        self.image_q: queue.Queue = queue.Queue()

        self.image_context: str = "This is a conversation between 2 people."
        self.img_cnt = 0
            
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

    def add_new_context_image(self, image):
        with self.has_new_image_cond:
            if self.img_cnt % 10 == 0: 
                self.image_q.put_nowait(image)
                self.has_new_image = True
                self.has_new_image_cond.notify_all()
            self.img_cnt += 1

    
    def context_image_start(self):
        """
        the interaction camera thread will call this method, which sends the
        context image to chat gpt which sends back a string describing the
        context of the situation 
        """
        image_client = OpenAI()
        while True:
            with self.has_new_image_cond:
                while not self.has_new_image and self.image_q.qsize() == 0:
                    self.has_new_image_cond.wait()
                    
                if self.image_q.qsize() == 0:
                    time.sleep(0.2)
                frame = self.image_q.get_nowait()
                self.logger.info("New image being processed")
                response = image_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text":"Describe the context, with focus on the background, in fewer than 10 words",
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
                self.image_context = additional_context
                self.logger.info("Context from image is: "+ additional_context)
                self.has_new_image=False
        # print(additional_context)
        # self.initialize_chat(additional_context + DEFAULT_CONTEXT)

    def keyword_input_handler(self, keyword_input: str):
        """
        the event is "keyboard_input", the keyboard thread calls this method
        once the user has inputted something. The assumption is that the
        voice rec thread has already added the user voice to self.voice_rec_q
        before this method is called, so after the keyboard input, the llama
        thread has all it needs to generate a response so it needs to wake up
        """
        with self.last_voice_rect_text_lock:
            try:
                # text_input = self.voice_rec_q.get_nowait()
                # print("text_input VOICEREC",text_input)
                # if text_input == "RESET":
                voice_input = self.last_voice_rec_text
            
                self.add_prompt_handler("A", generate_person_b_prompt(voice_input, keyword_input, self.image_context))

                
                print("REGEN", generate_person_b_prompt(voice_input, keyword_input))
                # else:
                    # self.add_prompt_handler
                        # "A", generate_person_b_prompt(text_input, keyword_input))
            except:
                self.logger.exception("keyword input failed")

    def voice_rec_input_handler(self, voice_rec_input: str):
        with self.last_voice_rect_text_lock:
            self.last_voice_rec_text = voice_rec_input
            self.add_prompt_handler("A", generate_person_b_prompt_no_keywords(voice_rec_input, self.image_context))

    def add_prompt_handler(self, sender: str, text: str):
        """
        There are 2 senders, A and B. A is the hearing person, B is the deaf person.
        """
        self.prompt_q.put_nowait(
            (sender, text))
        with self.need_response_cond:
            self.need_response = True
            self.need_response_cond.notify()

    def repeat_keywords_new_options_handler(self):
        self.add_prompt_handler("A", f"Please generate {NUMBER_OF_OPTIONS} more responses for person B each on its own line.")
        

    def start_conversation(self):
        try:
            while True:
                with self.need_response_cond:
                    while not self.need_response:
                        self.need_response_cond.wait()
                    sender, text = self.prompt_q.get_nowait()
                    assert sender == "A" or sender == "B"
                    self.logger.debug(f"New prompt is: {text}")

                    new_message = {
                        'role': 'user',
                        'content': text
                    }
                    self.messages.append(new_message)
                    print(new_message)
                    response_content = self.client.chat.completions.create(
                        model=self.model,
                        messages=self.messages
                    ).choices[0].message.content
                    self.messages.append({
                        "role": "assistant",
                        "content": response_content
                    })
                    print(response_content)
                    # self.logger.debug(f"llm message history:{self.messages}")
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
    # logger = logging.getLogger(__name__)
    # gpt = ChatGPTAPI(logger=logger, model="gpt-4o")
    # gpt_t = threading.Thread(target=gpt.start_conversation, daemon=True)
    # gpt_t.start()
    
    prompt = """PERSONAL STATEMENT: Please submit a brief statement of a maximum two paragraphs which addresses the following points. (Maximum size: 300 words/2400 characters.)  Your academic strengths and weaknesses.  Your objectives for participating in the project. * Your technical skills and relevant experience; i.e., machining, construction, laboratory instrumentation or techniques, programming

Acacdemic strength: gpa 4.15, major gpa 4.3, very hardworking. Has 2nd author publication for SpellRing, a ring that translates American Sign Languge Fingerspelling. Is working on co-first author publicication this semester relating to novel interface for 2 way communication between Deaf and hard of hearing people and hearing people. Strong in systems knowledge, has A+ in CS4414 systems programming with Ken birman. Also currently doing research with PhD Ben landrum, advised by Ken Birman, on making graph indices more efficicent. Skills include: familiarity with research process, python (data analysis), C++ """
    # gpt.add_prompt_handler("A", prompt)
    # time.sleep(10)

    client = OpenAI()

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    print(completion.choices[0].message.content)

    
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
