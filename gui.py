from functools import partial
import logging
import sys
from camera_module import OpenCVCamera
import linecache
from datetime import datetime
import os
import shutil
import argparse
import copy
import cv2
import numpy as np
import threading
import math
from typing import Callable, Tuple, Union
from voice_recognition import VoiceRecognition
import textwrap
from keyboard_gui import KeyboardGUIInput
from kbd_input import KeyboardInput
from video_recorder import VideoRecorder
from chatgpt_api import ChatGPTAPI
from android_input import AndroidInput
from tcp_server import TCPServer, IP, PORT
from tts import TTS, Emotions
from events import TCPServerEvents, AndroidInputEvents
import tracemalloc
import mp_drawing_utils
from gesture_recognition import GestureRecognition


class QueueHandler(logging.Handler):
    def __init__(self, add_new_msg_callback):
        super().__init__()

        self.add_new_msg_callback = add_new_msg_callback
    
    def emit(self, record):
        """
        Add the formatted log record to the queue.
        
        Args:
            record (LogRecord): The log record to process.
        """
        try:
            msg = self.format(record)
            self.add_new_msg_callback(msg)
        except Exception:
            self.handleError(record)


def delete_contents_folder(folder: str):
    """
    deletes all the files in a folder
    """
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


class VoiceRecTextComponent():
    def __init__(self,
                 text: str,
                 line_width: int,
                 bot_left: Tuple[int, int]):
        self.text = text
        self.bot_left = bot_left
        self.line_width = line_width
        self.text_list = textwrap.wrap(text, width=line_width)
        self.bg_color = (255, 0, 0)

    def render_component(self, canvas):
        font_scale = 1.0
        font_thickness = 2
        line_spacing = 10
        padding = 10

        # Calculate text dimensions
        max_width = 0
        total_height = 0
        for line in self.text_list:
            (text_width, text_height), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
            max_width = max(max_width, text_width)
            total_height += text_height + line_spacing

        # Calculate background rectangle dimensions
        rect_start = (self.bot_left[0], self.bot_left[1] - total_height - padding -10)
        rect_end = (self.bot_left[0] + max_width + 2*padding, self.bot_left[1])

        # Draw background rectangle
        cv2.rectangle(canvas, rect_start, rect_end, self.bg_color, -1)

        # Draw text
        y = self.bot_left[1] - padding
        for line in reversed(self.text_list):  # Reverse to start from bottom
            (_, text_height), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
            y -= text_height
            cv2.putText(canvas, line, (self.bot_left[0] + padding, y), 
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
            y -= line_spacing



class OptionComponent():
    def __init__(self,
                 text: str,
                 line_width: int,
                 bot_left: Tuple[int, int],
                 progress_100_callback: Callable,
                 progress_speed=1,
                 value = None):
        """
        bot_left means the bottom left of where the intended string is
        supposed to go, which is the first line of the wrapped string
        """
        self.value = value
        self.text = text
        self.text_list = textwrap.wrap(text, line_width)
        self.line_width = line_width
        self.thickness = 1
        self.bot_left_list = []

        self.text_width = 0
        self.text_height = 0
        self.line_heights = []
        self.event_subscribers: dict[str, dict[str, Callable]] = dict()
        self.color = (0, 255, 0)

        for i in range(len(self.text_list)):
            (width, height), baseline = cv2.getTextSize(
                self.text_list[i],
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                2
            )
            self.text_width = max(width, self.text_width)
            self.text_height += height + baseline
            self.line_heights.append(height + baseline)
            if i == 0:
                self.bot_left_list.append(bot_left)
            else:
                self.bot_left_list.append((
                    self.bot_left_list[i - 1][0],
                    self.bot_left_list[i-1][1] + height + baseline
                ))
        self.top_left = (bot_left[0], bot_left[1] - self.line_heights[0])
        self.bot_right = (self.top_left[0] + self.text_width,
                          self.top_left[1] + self.text_height)
        self.progress_speed = progress_speed
        self.selection_progress = 0

        # if is_selected == False then we decrease progress by
        # progress_speed until it reaches 0
        # if is_selected == True then we continously increase progress
        # by progress_speed since original_selection_time, until it hits 100
        self.is_selected = False
        self.is_hovered_over = False
        self.original_selection_time = 0

        self.register_event_subscriber(
            "progress_100", "gui", progress_100_callback)

    def render_component(self, canvas):
        if self.is_selected:
            self.bold()
        else:
            self.unbold()

        if self.is_hovered_over:
            self.bold()
            self.color =  (0, 0, 255)
        else:
            self.color =  (0, 255, 0)
            self.unbold()
        
        for i in range(len(self.text_list)):
            cv2.putText(
                canvas,
                self.text_list[i],
                self.bot_left_list[i],
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                self.color,
                self.thickness,
                cv2.LINE_AA)

        self.selection_progress = min(max(self.selection_progress, 0), 100)

        line_width = round(canvas.shape[1] * self.selection_progress / 100)
        line_width = max(line_width, 1)

        start_position = 0
        end_position = start_position + line_width
        end_position = min(end_position, canvas.shape[1])

        cv2.line(
            canvas,
            (start_position, self.top_left[1]),
            # End at the calculated right position
            (end_position, self.top_left[1]),
            self.color,
            thickness=3
        )

        cv2.line(
            canvas,
            # Start at the calculated bottom left
            (start_position, self.bot_right[1] + 5),
            # End at the calculated bottom right
            (end_position, self.bot_right[1] + 5),
            self.color,
            thickness=3
        )

    def register_event_subscriber(self, event_name: str, subscriber_name: str, fn: Callable):
        if event_name not in self.event_subscribers:
            self.event_subscribers[event_name] = dict()
        self.event_subscribers[event_name][subscriber_name] = fn

    def notify_event_subscriber(self, event_name: str, subsciber_name: str, *args):
        self.event_subscribers[event_name][subsciber_name](*args)

    def update_progress(self):
        """
        if self.is_selected == False, then we decrease selection_progess
        else we increase selection_progress
        """
        if not self.is_selected:
            self.selection_progress = max(0,
                                          self.selection_progress - 2 * self.progress_speed)
        else:
            self.selection_progress = min(100,
                                          self.selection_progress + self.progress_speed)
        if self.selection_progress == 100:
            event_name = "progress_100"
            if event_name in self.event_subscribers:
                for sub in self.event_subscribers[event_name].keys():
                    self.notify_event_subscriber(event_name, sub)

    def bold(self):
        self.thickness = 2

    def unbold(self):
        self.thickness = 1
    def select(self):
        self.is_selected = True
    def unselect(self):
        self.is_selected = False

    def hover(self):
        self.is_hovered_over = True
    def unhover(self):
        self.is_hovered_over = False
        
    def reset(self):
        self.selection_progress = 0
        self.unbold()
        self.unselect()
        self.unhover()


def _normalized_to_pixel_coordinates(normalized_x: float,
                                     normalized_y: float,
                                     image_width: int,
                                     image_height: int) -> Union[None, Tuple[int, int]]:
    """Converts normalized value pair to pixel coordinates."""

    # Checks if the float value is between 0 and 1.
    def is_valid_normalized_value(value: float) -> bool:
        return (value > 0 or math.isclose(0, value)) and (value < 1 or
                                                          math.isclose(1, value))

    if not (is_valid_normalized_value(normalized_x) and
            is_valid_normalized_value(normalized_y)):
        # TODO: Draw coordinates even if it's outside of the image bounds.
        return None
    x_px = min(math.floor(normalized_x * image_width), image_width - 1)
    y_px = min(math.floor(normalized_y * image_height), image_height - 1)
    return x_px, y_px


class GUIClass():
    """
    The GUI class is responsible for displaying the GUI.

    As this is simulating an AR app, all gui objects will be drawn
    onto the frame recieved by the camera.

    """

    def __init__(self,
                 exit_event: threading.Event,
                 hand_cam: OpenCVCamera,
                 logger: logging.Logger,
                 interaction_cam = None,
                 fullscreen=False,
                 black=False,
                 emotional_voice=False
                 ) -> None:
        self.logger = logger
        self.emotional_voice = emotional_voice
        
        self.interaction_cam = interaction_cam
        self.hand_cam = hand_cam
        if self.interaction_cam is not None and self.hand_cam is None:
            print("must have hand_cam")
            sys.exit(0)
        self.num_cam = 0
        if self.interaction_cam is not None:
            self.num_cam += 1
        if self.hand_cam is not None:
            self.num_cam += 1

        if self.num_cam != 1 and self.num_cam != 2:
            print("must have either 1 or 2 cameras")
            sys.exit()
        # self.interaction_cam = OpenCVCamera(0, 480, 640, 30.0)
        # self.hand_cam = OpenCVCamera(4, 720, 1280, 30.0)

        if self.num_cam == 1:
            self.frame_width = int(self.hand_cam.cam_width)
            self.frame_height = int(self.hand_cam.cam_height)
        elif self.num_cam == 2:
            self.frame_width = int(self.interaction_cam.cam_width)
            self.frame_height = int(self.interaction_cam.cam_height)
    
        self.black = black
        
        self.fullscreen = fullscreen
        # list of the subscribers (str) along with their methods.
        # when an event of interest occurs, the gui object will
        # call the corresponding method of the subscriber
        self.subscribers: dict[str, Callable] = dict()

        # the order by which to render the elements.
        self.render_order: list[str] = []

        # key is name : str, value is function to draw on canvas
        self.render_functions: dict[str, Callable] = dict()
        # whether the thing to render is ready to be rendered
        # (call the function in render_functions).
        # key is str, value is bool
        self.render_ready: dict[str, bool] = dict()

        # canvas to draw everything on, in the render method,
        # this should be updated as the camera read frames.
        self.canvas = np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8)
        # self.frame_height = -1
        # self.frame_width = -1

        # if exit_event is set then must end.
        self.exit_event = exit_event

        # more robust is a State Machine to transition between states
        # right now, the render_order and render_ready are used to determine
        # what the current state is

        # variables for llm rendering
        self.llm_options: list[OptionComponent] = []
        self.selected_llm_option_index = -1
        self.number_of_llm_options = 5
        if self.emotional_voice: 
            self.emotion_options: list[OptionComponent] = [
                OptionComponent("SAD",
                                50,
                                (100, 100),
                                self.emotion_progress_full_handler, value=Emotions.SAD),

                OptionComponent("HAPPY",
                                50,
                                (100, 200),
                                self.emotion_progress_full_handler, value=Emotions.HAPPY
                                ),
                OptionComponent("NEUTRAL",
                                50,
                                (100, 300),
                                self.emotion_progress_full_handler, value=Emotions.NEUTRAL),

                OptionComponent("TERRIFIED",
                                50,
                                (100, 400),
                                self.emotion_progress_full_handler, value=Emotions.TERRIFIED),

                OptionComponent("ANGRY",
                                50,
                                (100, 500),
                                self.emotion_progress_full_handler, value=Emotions.ANGRY)
            ]

            self.selected_emotion_option_index = -1
            self.number_of_emotion_options = 5
        else:
            self.selected_emotion_option_index = -1
            self.number_of_emotion_options = 0
            self.emotion_options = []
        
        self.voice_rec_text_component: VoiceRecTextComponent = VoiceRecTextComponent(
            "", 36, (0, 0))


    def register_subscriber(self, subscriber_name: str, fn: Callable):
        self.subscribers[subscriber_name] = fn

    def notify_subscriber(self, subsciber_name: str, *args):
        self.subscribers[subsciber_name](*args)

    def add_ui_component(self,
                         component_name: str,
                         component_render_fn: Callable):
        self.render_order.append(component_name)
        self.render_functions[component_name] = component_render_fn
        self.render_ready[component_name] = False

    def update_canvas(self, canvas: np.ndarray):
        """
        draws everything on specified canvas
        """
        for ui_element in self.render_order:
            if self.render_ready[ui_element]:
                self.render_functions[ui_element](canvas)
        

    def handle_input(self):
        pressed_key = cv2.waitKeyEx(1) & 0xFF
        if pressed_key != 255:
            print(pressed_key)

        if pressed_key == 27:
            self.exit_event.set()

            # there is probably a better way to do this
            # this is letting the subsciber handle the exit
            for subscriber in list(self.subscribers.keys()):
                if "-exit" in subscriber:
                    print("SUBCRIBRE TRYING TO EXIT IS", subscriber)
                    self.notify_subscriber(subscriber)
            print("esc done")
        elif pressed_key == ord(' '):
            self.notify_subscriber("voice-start")
        elif pressed_key == 84:  # up arrow:
            if self.render_ready['llm-options']:
                self.selected_llm_option_index = (
                    self.selected_llm_option_index + 1) % self.number_of_llm_options
        elif pressed_key == 82:  # down arrow
            if self.render_ready['llm-options']:
                self.selected_llm_option_index = (
                    self.selected_llm_option_index - 1) % self.number_of_llm_options
        elif pressed_key == 13:  # enter key
            pass

    def render(self):
        self.logger.debug("start rendering program")
        while True:
            if self.num_cam == 2:
                ret_1, hand_frame = self.hand_cam.read()
                ret_2, interaction_frame = self.interaction_cam.read()
                if not ret_1 or not ret_2:
                    print("can't parse frame")
                    break
            elif self.num_cam == 1:
                ret, hand_frame = self.hand_cam.read()
                if not ret:
                    print("can't parse frame")
                    break

            if hand_frame.shape[0] != self.frame_height or hand_frame.shape[1] != self.frame_width:
                # hand_frame = cv2.resize(hand_frame, (self.frame_width, self.frame_height), interpolation=cv2.INTER_LINEAR)
                pass
            if self.num_cam == 2:
                if interaction_frame.shape[0] != self.frame_height or interaction_frame.shape[1] != self.frame_width:
                    interaction_frame = cv2.resize(interaction_frame, (self.frame_width, self.frame_height), interpolation=cv2.INTER_LINEAR)
            
            if self.black:
                self.canvas = np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8)
            else:
                if self.num_cam == 1:
                    self.canvas = hand_frame
                elif self.num_cam == 2:
                    self.canvas = interaction_frame

            # self.notify_subscriber("new-frame-to-process", cv2.flip(hand_frame, 1))
            self.notify_subscriber("new-frame-to-process", hand_frame)
            self.update_canvas(self.canvas)
            if self.black:
                if self.num_cam == 2:
                    self.update_canvas(interaction_frame)
                    self.notify_subscriber(
                        "new-frame-to-record", interaction_frame)
                else:
                    self.update_canvas(hand_frame)
                    self.notify_subscriber(
                        "new-frame-to-record", hand_frame)
            else:
                self.notify_subscriber(
                    "new-frame-to-record", self.canvas)
            self.handle_input()
            if self.fullscreen:
                cv2.namedWindow("realtime llm", cv2.WND_PROP_FULLSCREEN)
                cv2.setWindowProperty(
                    "realtime llm", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            else:
                cv2.namedWindow("realtime llm")
                pass
            cv2.imshow("realtime llm", cv2.resize(
                (self.canvas), (1920, 1080), interpolation=cv2.INTER_LINEAR))
            if self.exit_event.is_set():
                break
            
    def render_mediapipe(self, canvas, draw_bounding_rectangle, brect, hand_sign, landmark_list):
        mp_drawing_utils.draw_gesture_and_landmarks_on_image(
        canvas,
        draw_bounding_rectangle,
        brect,
        hand_sign,
        landmark_list)
        cv2.putText(canvas, f"Number:{hand_sign}", (canvas.shape[1] - 250, canvas.shape[0]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
    def mediapipe_callback_handler(self, draw_bounding_rectangle, brect, hand_sign, landmark_list):
        # self.logger.debug(f"mediapipe classify user gestures: {hand_sign}")
        
        self.render_functions["mediapipe"] = lambda canvas: self.render_mediapipe(
            canvas,
            draw_bounding_rectangle,
            brect,
            hand_sign,
            landmark_list)

        self.render_ready["mediapipe"] = True
        if hand_sign == "OK":
            if self.render_ready["emotion_options"] and self.selected_emotion_option_index >=0 and self.selected_emotion_option_index < self.number_of_emotion_options:
                # print(hand_sign)
                # self.logger.debug(f"user is selecting emotion index:  {self.selected_emotion_option_index}")
                # self.emotion_options[self.selected_emotion_option_index].select()
                # self.emotion_options[self.selected_emotion_option_index].update_progress()
                for i in range(self.number_of_emotion_options):
                    if i == self.selected_emotion_option_index:
                        self.emotion_options[i].select()
                    else:
                        self.emotion_options[i].unselect()
            if self.render_ready["llm-options"] and self.selected_llm_option_index >=0 and self.selected_llm_option_index < self.number_of_llm_options:
                # self.logger.debug(f"user is selecting llm index:  {self.selected_emotion_option_index}")
                for i in range(self.number_of_llm_options):
                    if i == self.selected_llm_option_index:
                        self.llm_options[i].select()
                    else:
                        self.llm_options[i].unselect()

        else:
            for i in range(len(self.llm_options)):
                self.llm_options[i].unselect()
            for i in range(self.number_of_emotion_options):
                self.emotion_options[i].unselect()
            if self.render_ready["llm-options"]:
                self.update_selected_llm_option_mp(hand_sign)
            if self.render_ready["emotion_options"]:
                self.update_selected_emotion_option_mp(hand_sign)
        
    ## LLM OPTIONS CODE SECTION
    def update_selected_llm_option_mp(self, hand_sign: str):
        self.selected_llm_option_index = int(hand_sign) - 1
        for i in range(self.number_of_llm_options):
            if i == self.selected_llm_option_index:
                self.llm_options[i].hover()
            else:
                self.llm_options[i].unhover()

            
    def render_llm_options(self, canvas):
        for i in range(len(self.llm_options)):
            # if i == self.selected_llm_option_index:
                # self.llm_options[i].is_selected = True
                # self.llm_options[i].color = (0, 0, 255)
            # else:
                # self.llm_options[i].is_selected = False
                # self.llm_options[i].color = (0, 255, 0)

            self.llm_options[i].update_progress()
            self.llm_options[i].render_component(canvas)

    def llm_response_handler(self, response: str):
        print("response recieved by gui thread")
        line_width = 36
        llm_options = [rep for rep in response.splitlines() if len(
            rep) > 2 and rep[0].isdigit()]
        self.logger.info(f"ChatGPT: {llm_options}")
        assert(len(llm_options) == self.number_of_llm_options - 1)
        self.llm_options.clear()
        
        for i in range(len(llm_options)):
            if i == 0:
                text_bottom_left = (self.frame_width//2, 50 + 100 * i)
            else:
                text_bottom_left = (
                    self.llm_options[i - 1].bot_left_list[-1][0],
                    self.llm_options[i - 1].bot_left_list[-1][1] + 2 * self.llm_options[i - 1].line_heights[-1])
            self.llm_options.append(
                OptionComponent(
                    llm_options[i],
                    line_width,
                    text_bottom_left, self.llm_progress_full_handler))
        reset_option_bottom_left = (
            self.llm_options[-1].bot_left_list[-1][0],
            self.llm_options[-1].bot_left_list[-1][1] + 2 * self.llm_options[-1].line_heights[-1])
        self.llm_options.append(
            OptionComponent(
                "RESET",
                line_width,
                reset_option_bottom_left,
                self.llm_reset_progress_full_handler))

        self.render_ready["llm-options"] = True

        self.render_ready["voice_rec_text"] = False

    def llm_reset_progress_full_handler(self):
        if self.render_ready['llm-options']:
            self.logger.info("reset chose n   bb")
            self.notify_subscriber(
                "llm-reset-keywords"
            )
            self.notify_subscriber("android-need-input")
            
            # if self.selected_llm_option_index != 4:
            #     self.notify_subscriber(
            #         "llm-add-prompt",
            #         "B",
            #         self.llm_options[self.selected_llm_option_index].text
            #     )
            #     self.notify_subscriber(
            #         "done_choosing_llm_option",
            #         copy.deepcopy(self.llm_options[self.selected_llm_option_index].text))
            #     self.selected_llm_option_index = -1
            self.selected_llm_option_index = -1
            self.render_ready["llm-options"] = False
            self.render_ready["keyword_text"] = False
            # self.render_ready["emotion_options"] = True
            # else:
            #     pass
            
    def llm_progress_full_handler(self):
        if self.render_ready['llm-options']:
            self.logger.info(f"DHH: {self.llm_options[self.selected_llm_option_index].text}")
            self.notify_subscriber(
                "llm-add-prompt",
                "B",
                self.llm_options[self.selected_llm_option_index].text
            )
            self.render_ready["llm-options"] = False
            self.render_ready["keyword_text"] = False
            if self.emotional_voice: 
                self.render_ready["emotion_options"] = True
                self.notify_subscriber(
                    "done_choosing_llm_option", copy.deepcopy(self.llm_options[self.selected_llm_option_index].text))
            else:
                self.notify_subscriber(
                    "done_choosing_llm_option_neutral",
                    copy.deepcopy(self.llm_options[self.selected_llm_option_index].text)
                )

            self.selected_llm_option_index = -1


    def render_tts_indicator(self, canvas):
        cv2.putText(
            canvas,
            "TTS is speaking",
            (200, 100),
            cv2.FONT_HERSHEY_COMPLEX,
            1.0,
            (255, 0, 0),
            1)

            
    def render_emotion_options(self, canvas):
        for i in range(self.number_of_emotion_options):
            # if i == self.selected_emotion_option_index:
                # self.emotion_options[i].is_selected = True
                # self.emotion_options[i].color = (0, 0, 255)
            # else:
                # self.emotion_options[i].is_selected = False
                # self.emotion_options[i].color = (0, 255, 0)

            self.emotion_options[i].update_progress()
            self.emotion_options[i].render_component(canvas)


    def new_keyword_handler(self, text):
        self.render_functions["keyword_text"] = lambda canvas: self.render_keyword_text(canvas, text)
        self.render_ready["keyword_text"] = True
        
    def render_keyword_text(self, canvas, text):
        cv2.putText(canvas, text, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 0), 2)
            
    def emotion_progress_full_handler(self):
        if self.render_ready['emotion_options']:
            self.logger.info(f"the user chooses emotion option: {self.emotion_options[self.selected_emotion_option_index].value}")
            self.notify_subscriber(
                "done_choosing_emotion_option",
                self.emotion_options[self.selected_emotion_option_index].value
            )
            self.render_ready["emotion_options"] = False
            self.selected_emotion_option_index = -1
            for i in range(self.number_of_emotion_options):
                self.emotion_options[i].reset()
            

    def update_selected_emotion_option_mp(self, hand_sign: str):
        self.selected_emotion_option_index = int(hand_sign) - 1
        for i in range(self.number_of_emotion_options):
            if i == self.selected_emotion_option_index:
                self.emotion_options[i].hover()
            else:
                self.emotion_options[i].unhover()

    
    def voice_input_ready_handler(self, text: str):
        self.logger.info(f"Hearing: {text}")
        self.render_ready["voice_rec_text"] = True
        self.voice_rec_text_component = VoiceRecTextComponent(
            text, 25, (self.canvas.shape[1] // 2 - 100, self.canvas.shape[0] - 10))
        self.render_functions["voice_rec_text"] = lambda canvas: self.voice_rec_text_component.render_component(
            canvas)
        
    def on_voice_recording_start(self):
        """
        Shows Recording Voice on screen to indicate that the hearing person's voice has been
        picked up
        """
        self.render_ready["voice-recording-start"] = True
        self.render_functions["voice-recording-start"] = lambda canvas: cv2.putText(
            canvas,
            "Recording Voice",
            (100, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            2,
            cv2.LINE_AA)

    def on_voice_recording_stop(self):
        """
        removes the Recording Voice text on screen
        """
        self.render_ready["voice-recording-start"] = False

    def on_tts_start_handler(self):
        self.render_ready["tts_indicator"] = True

    def on_tts_end_handler(self):
        self.render_ready["tts_indicator"] = False

def display_top(snapshot, key_type='lineno', limit=15):
    snapshot = snapshot.filter_traces((
        tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
        tracemalloc.Filter(False, "<unknown>"),
    ))
    top_stats = snapshot.statistics(key_type)

    print("Top %s lines" % limit)
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        # replace "/path/to/module/file.py" with "module/file.py"
        filename = os.sep.join(frame.filename.split(os.sep)[-2:])
        print("#%s: %s:%s: %.1f KiB"
              % (index, filename, frame.lineno, stat.size / 1024))
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            print('    %s' % line)

    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        print("%s other: %.1f KiB" % (len(other), size / 1024))
    total = sum(stat.size for stat in top_stats)
    print("Total allocated size: %.1f KiB" % (total / 1024))

    
if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    
    tracemalloc.start()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--path",
        help="path to output the data like video, voice recording,...",
        required=True)
    parser.add_argument(
        "-fs",
        "--fullscreen",
        help="whether to display the gui in fullscreen",
        action='store_true')
    parser.add_argument(
        "-wi",
        "--word-input",
        help="whether to use the keyboard or android phone to input keywords",
        choices = ["keyboard", "android"],
        default = 'android',
        nargs = "?",
        const = "android"
    )
    parser.add_argument(
        "-b",
        "--black",
        help="whether to black out the screen to display on xreal glasses or not",
        action="store_true"
    )
    parser.add_argument(
        "-vg",
        "--voice-gender",
        help="gender",
        choices=["female","male"],
        default="male"
    )
    parser.add_argument(
        "-d",
        "--delete",
        help="whether to delete the folder specified or not",
        action="store_true"
    )
    parser.add_argument(
        "-oi",
        "--output_index",
        help="audio output index, based on python -m sounddevice"
    )
    parser.add_argument(
        "-ii",
        "--input_index",
        help="audio input index, based on list_audio_indices.py"
    )
    parser.add_argument(
        "-l",
        "--level",
        help = "select level. Level 1 means 1 sign, level 2 means 3 signs, level 3 means full sentence",
        choices = ["1", "2", "3"],
        default = "1"
    )

    parser.add_argument(
        "-em",
        "--emotional-voice",
        help = "whether to use emotional voice or not",
        action = "store_true"
    )
    

    args = parser.parse_args()

    folder_path = ""
    video_path = ""

    if os.path.exists(args.path):
        if not args.delete:
            print(
                f"{args.path} already exists, do you want to delete the contents or no [y/n]")
            while True:
                delete_or_not = input()
                if delete_or_not == "y":
                    delete_contents_folder(args.path)
                    os.mkdir(os.path.join(args.path, "video"))
                    break
                elif delete_or_not == "n":
                    break
                else:
                    print("please only enter y or n:")
        else:
            delete_contents_folder(args.path)
            os.mkdir(os.path.join(args.path, "video"))
                
    else:
        os.mkdir(args.path)
        os.mkdir(os.path.join(args.path, "video"))



    ### initialize the files and folders
    folder_path = args.path
    video_path = os.path.join(args.path, "video")
    time_rn = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
    video_filename = f"video_{time_rn}.mp4"

    # interaction log will log the non technical details of the interaction
    # will log
    # - what the llm choices are,
    # - what the DHH person chooses,
    # - what the hearing person says (voice rec)
    # - what the interpreter sends
    interaction_handler = logging.FileHandler(os.path.join(folder_path, f'interaction_log_{time_rn}.txt'))
    interaction_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s-%(levelname)s-%(message)s')

    interaction_handler.setFormatter(formatter)
    logger.addHandler(interaction_handler)

    # debug log logs everything, when the threads recieve what, .... It is a superset of the
    # interaction log
    debug_handler = logging.FileHandler(os.path.join(folder_path, f'debug_log_{time_rn}.txt'))
    debug_handler.setLevel(logging.DEBUG)
    debug_handler.setFormatter(formatter)
    logger.addHandler(debug_handler)


    # socket_handler = PlainTextTcpHandler(host=IP, port=PORT)
    # socket_handler.setLevel(logging.INFO)
    # logger.addHandler(socket_handler)
    
    



    ### start declaring the different components


    exit_event: threading.Event = threading.Event()
    hand_cam = OpenCVCamera(0, 720, 1280, 30.0)
    interaction_cam = OpenCVCamera(2, 720, 1280, 30.0)

    gui: GUIClass = GUIClass(exit_event=exit_event,
                             fullscreen=args.fullscreen,
                             black=args.black,
                             hand_cam = hand_cam,
                             interaction_cam=interaction_cam,
                             logger=logger,
                             emotional_voice=args.emotional_voice
                             )
    video_recorder = VideoRecorder(video_path, video_filename, frame_size=(gui.frame_width, gui.frame_height))
    voice_rec: VoiceRecognition = VoiceRecognition(
        input_device_index=args.input_index,
        logger=logger)

    llm: ChatGPTAPI = ChatGPTAPI(logger=logger)

    def tts_finished_speaking_handler():
        voice_rec.voice_start_handler()
        gui.on_tts_end_handler()

    tts: TTS = TTS(
        tts_start_handler=gui.on_tts_start_handler,
        tts_end_handler=tts_finished_speaking_handler,
        gender=args.voice_gender,
        output_device_index=args.output_index,
        logger=logger)
    
    gesture_rec: GestureRecognition = GestureRecognition(
        logger=logger)
    
    if args.word_input == "keyboard":
        keyboard_input = KeyboardGUIInput(
            new_keyboard_input_handler_list=[
                llm.keyword_input_handler, gui.new_keyword_handler],
            logger=logger)
        

        # keyboard_input.register_event_subscriber("keyboard_input_ready",
                                                 # "llm",
                                                 # llm.keyword_input_handler)

                
        keyboard_input_log_handler = QueueHandler(
            keyboard_input.new_message_handler)
        
        formatter = logging.Formatter('%(message)s\n')
        keyboard_input_log_handler.setFormatter(formatter)
        keyboard_input_log_handler.setLevel(logging.INFO)
        logger.addHandler(keyboard_input_log_handler)
        
        voice_rec.register_event_subscriber(
            "voice_input_ready",
            "keyword_input",
            keyboard_input.need_keyboard_input)

        keyboard_input_thread: threading.Thread = threading.Thread(
            target=keyboard_input.run,
            daemon=True
        )
        keyboard_input_thread.start()
    elif args.word_input == "android":
        tcp_serv = TCPServer(logger=logger)
        android_input = AndroidInput(logger=logger)
        tcp_serv.register_event_subscriber(
            TCPServerEvents.MSG_RECEIVED,
            "android_input",
            android_input.server_msg_received_handler
        )

        tcp_serv_thread = threading.Thread(target=tcp_serv.start_server, daemon=True)
        tcp_serv_thread.start()
        android_input.register_event_subscriber(AndroidInputEvents.INPUT_READY,
                                                "llm",
                                                llm.keyword_input_handler)
        android_input.register_event_subscriber(AndroidInputEvents.INPUT_READY,
                                                "gui",
                                                gui.new_keyword_handler)
        voice_rec.register_event_subscriber("voice_input_ready",
                                            "keyword_input",
                                            android_input.need_input_handler)
        android_input_thread: threading.Thread = threading.Thread(
            target=android_input.start_input,
            daemon=True
        )
        android_input_thread.start()
        gui.register_subscriber("android-need-input",
                                android_input.need_input_handler)
        
        tcp_server_handler = QueueHandler(tcp_serv.new_msg_to_write_handler)
        formatter = logging.Formatter('%(message)s\n')
        tcp_server_handler.setFormatter(formatter)
        tcp_server_handler.setLevel(logging.INFO)
        # logger.addHandler(tcp_server_handler)

    if gui.emotional_voice:
        gui.register_subscriber("done_choosing_llm_option", tts.add_text_handler)
        gui.register_subscriber(
            "done_choosing_emotion_option",
            tts.add_emotion_handler
        )
    else:
        gui.register_subscriber("done_choosing_llm_option_neutral", partial(tts.add_tts_handler, Emotions.NEUTRAL))

    
    gui.register_subscriber("llm-reset-keywords", lambda : llm.voice_rec_input_handler("RESET"))
    gui.register_subscriber("voice-start", voice_rec.voice_start_handler)
    gui.register_subscriber("llm-add-prompt", llm.add_prompt_handler)
    gui.register_subscriber("new-frame-to-record",
                            video_recorder.new_frame_event_handler)
    gui.register_subscriber("new-frame-to-process",
                            gesture_rec.new_frame_handler)
    gui.register_subscriber("video-record-exit",
                           video_recorder.exit_event_handler)


    voice_rec.register_event_subscriber("voice_input_ready",
                                        "llm",
                                        llm.voice_rec_input_handler)
    
    voice_rec.register_event_subscriber("voice_input_ready",
                                        "gui",
                                        gui.voice_input_ready_handler)
    
    
    llm.register_event_subscriber(
        "new-response",
        "gui",
        gui.llm_response_handler
    )

    gesture_rec.register_event_subscriber("finished_processing_frame",
                                          "gui",
                                          gui.mediapipe_callback_handler)

    gui.add_ui_component("llm-options", gui.render_llm_options)
    gui.add_ui_component("voice_rec_text", lambda x: x)
    gui.add_ui_component("keyword_text", lambda x : x)
    gui.add_ui_component("voice_recording_start", lambda x: x)
    gui.add_ui_component("emotion_options", gui.render_emotion_options)
    gui.add_ui_component("mediapipe", lambda: None)

        
    gui.add_ui_component("tts_indicator", gui.render_tts_indicator)


    voice_rec_thread: threading.Thread = threading.Thread(
        target=voice_rec.start_voice_input,
        daemon=True
    )
    llm_thread: threading.Thread = threading.Thread(
        target=llm.start_conversation,
        daemon=True
    )
    
    video_recorder_thread: threading.Thread = threading.Thread(
        target=video_recorder.write_video
    )
    tts_thread: threading.Thread = threading.Thread(
        target=tts.start_tts,
        daemon=True
    )
    gesture_rec_thread: threading.Thread = threading.Thread(
        target = gesture_rec.start_recognition,
        daemon=True
    )

    logger.info(f"Level: {args.level}")
    gesture_rec_thread.start()
    voice_rec_thread.start()
    llm_thread.start()
    video_recorder_thread.start()
    tts_thread.start()
    gui.render()
    
    video_recorder_thread.join()
    snapshot = tracemalloc.take_snapshot()
    display_top(snapshot)

    
