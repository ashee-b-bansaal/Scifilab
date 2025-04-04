import logging
import sys
from camera_module import OpenCVCamera
import linecache
from datetime import datetime
import time
import os
import shutil
import copy
import cv2
import numpy as np
import threading
import math
from typing import Callable, Tuple, Union
import textwrap
import tracemalloc
import mp_drawing_utils
from tts import Emotions



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

def flatten(xss):
    return [x for xs in xss for x in xs]

class TextComponent():
    def __init__(self,
                 text: str,
                 line_width: int,
                 bot_left: Tuple[int, int],
                 progress_bar: bool = False,
                 full_progress_time: float = 1.0, #progress will start counting once the compoent is rendered
                 full_progress_width: int = 0,
                 bg_color=(255, 0, 0),
                 font_scale: float = 0.8,
                 font_thickness: int = 2,
                 ): # number of pixels
        self.font_scale = font_scale
        self.font_thickness = font_thickness
        self.text = text
        self.bot_left = bot_left
        self.line_width = line_width

        self.text_list = self.text.split('\n')
        self.text_list = flatten([textwrap.wrap(line, width=line_width) for line in self.text_list])
        print("text list for text component is ", self.text_list) 
        
        self.bg_color = bg_color

        self.progress_bar = progress_bar
        self.full_progress_time = full_progress_time
        self.current_progress_time = 0.0
        self.start_progress_timestamp = 0.0
        self.full_progress_width = full_progress_width
        self.current_progress_bar_width = 0

    def set_bot_left(self, bot_left):
        self.bot_left = bot_left
    def set_full_progress_time(self, full_progress_time):
        self.bot_left = full_progress_time

    
    def set_text(self, text):
        self.text = text
        self.text_list = self.text.split('\n')
        self.text_list = flatten([textwrap.wrap(line, width=self.line_width) for line in self.text_list])

    def reset_progress(self):
        self.current_progress_time = 0.0

    def set_full_progress_time(self, length: float):
        self.full_progress_time = length
        
    def render_component(self, canvas):
            
        font_scale = self.font_scale
        font_thickness = self.font_thickness
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
        self.rect_start = (self.bot_left[0], self.bot_left[1] - total_height - padding -10)
        self.rect_end = (self.bot_left[0] + max_width + 2*padding, self.bot_left[1])

        # Draw background rectangle
        cv2.rectangle(canvas, self.rect_start, self.rect_end, self.bg_color, -1)

        # Draw text
        y = self.bot_left[1]
        for line in reversed(self.text_list):  # Reverse to start from bottom
            (_, text_height), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
            y -= text_height
            cv2.putText(canvas, line, (self.bot_left[0] + padding, y), 
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
            y -= line_spacing

        if self.progress_bar:
            if self.current_progress_time == 0.0:
                self.start_progress_timestamp = time.time()
            self.current_progress_time = time.time() - self.start_progress_timestamp
            if self.current_progress_time < self.full_progress_time:
                self.current_progress_bar_width = int((self.current_progress_time / self.full_progress_time) * self.full_progress_width)

            cv2.line(
                canvas,
                (0, self.rect_start[1] - 5),
                # End at the calculated right position
                (self.current_progress_bar_width, self.rect_start[1] - 5),
                (255, 0, 0),
                thickness=3
            )
            
            cv2.line(
                canvas,
                # Start at the calculated bottom left
                (0, self.rect_end[1]),
                # End at the calculated bottom right
                (self.current_progress_bar_width, self.rect_end[1]),
                (255, 0, 0),
                thickness=3
            )

class OptionComponent():
    def __init__(self,
                 text: str,
                 line_width: int,
                 bot_left: Tuple[int, int],
                 progress_100_callback: Callable,
                 progress_speed=2,
                 show_bg=True,
                 value = None):
        """
        bot_left means the bottom left of where the intended string is
        supposed to go, which is the first line of the wrapped string
        """
        self.show_bg = show_bg
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
        self.color = (255, 255, 255)

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
            self.color =  (255, 255, 255)
            self.unbold()
        
        for i in range(len(self.text_list)):
            (text_width, text_height), _ = cv2.getTextSize(self.text_list[i], cv2.FONT_HERSHEY_SIMPLEX, 1.3, self.thickness)
            rect_start = (self.bot_left_list[i][0], self.bot_left_list[i][1] - text_height - 5)
            rect_end = (self.bot_left_list[i][0] + text_width, self.bot_left_list[i][1] + 10)

            if self.show_bg:
                cv2.rectangle(
                    canvas,
                    rect_start,
                    rect_end,
                    (0, 0, 0),
                    -1
                )
            cv2.putText(
                canvas,
                self.text_list[i],
                self.bot_left_list[i],
                cv2.FONT_HERSHEY_SIMPLEX,
                1.3,
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
                 emotional_voice=False,
                 keyword_text=False,
                 tasks: list[str] = [],
                 tutorial: bool=False,
                 image_context: bool=False) -> None:
    
    
        self.tasks = tasks
        self.current_task_index_lock: threading.Lock = threading.Lock()
        self.current_task_index = 0
        self.image_context = image_context
        self.image_context_time_interval = 15.0
        
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

        self.tutorial = tutorial
        if self.tutorial:
            self.tutorial_options: list[OptionComponent] = []
            
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

        self.voice_rec_text_component: TextComponent = TextComponent(
            "", 36, (0, 0))

        self.keyword_text = keyword_text
        if self.keyword_text:
            self.keyword_text_component: TextComponent = TextComponent(
                "",
                30,
                (50, 75),
                bg_color=(255, 255, 255)
            )

        self.tts_indicator_component: TextComponent = TextComponent(
            "Speaking",
            30,
            (50, 75),
            True,
            2.0,
            self.frame_width,
            bg_color=(0, 165, 255)
        )
        self.last_chosen_llm_option_text = ""
        self.last_chosen_llm_option_bot_left = (0, 0)
        self.has_tasks = len(self.tasks) > 0
        if self.has_tasks:
            font_scale = 0.6
            bot_left = (10, 390)
            bg_color = (0, 0, 0)
    
            if self.tutorial:
                font_scale = 1.5
                bot_left = (10, 200)
                bg_color = (255, 0, 0)
            self.current_task_component = TextComponent(
                self.tasks[0],
                30,
                bot_left,
                False,
                bg_color=bg_color,
                font_scale=font_scale,
                font_thickness=2
            )
        speaker_image_path = "speaker_smaller.png"
        self.speaker_icon = cv2.resize(cv2.imread(speaker_image_path), (50, 50))
        if self.image_context:
            self.last_image_context_time = time.time()



        # llm option recieved time - voice rec recieve time = delay from llm for when 

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
            if not self.tutorial:
                self.notify_subscriber("voice-start")
                self.render_ready["current_task"] = True
                newline = "\n"
                if self.has_tasks:
                    self.logger.info(
                        f"Phase {self.current_task_index}: {self.tasks[self.current_task_index].replace(newline, ',')}")
            else:
                self.render_ready["llm-options"] = True
                self.render_ready["current_task"] = True
                newline = "\n"
                if self.has_tasks:
                    self.logger.info(
                        f"Phase {self.current_task_index}: {self.tasks[self.current_task_index].replace(newline, ',')}")
                
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
                if self.image_context and time.time() - self.last_image_context_time > self.image_context_time_interval:
                    self.last_image_context_time = time.time()
                    self.logger.debug(str(self.last_image_context_time))
                    self.notify_subscriber("new-image-context", interaction_frame)
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
            # self.render_ready["mediapipe"] = False
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
        cv2.putText(canvas, f"Sign:{hand_sign}", (canvas.shape[1] - 250, canvas.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
    def mediapipe_callback_handler(self, draw_bounding_rectangle, brect, hand_sign, landmark_list, handedness):
        # self.logger.debug(f"mediapipe classify user gestures: {hand_sign}")
        # if hand_sign == "0":
            # self.render_ready["mediapipe"] = False
            # return
        if handedness.classification[0].label == "Left": 
            # print(handedness)
            self.render_functions["mediapipe"] = lambda canvas: self.render_mediapipe(
                canvas,
                draw_bounding_rectangle,
                brect,
                hand_sign,
                landmark_list)

            self.render_ready["mediapipe"] = True
            if hand_sign == "0":
                return
            if hand_sign == "OK":
                if self.render_ready["emotion_options"] and self.selected_emotion_option_index >=0 and self.selected_emotion_option_index < self.number_of_emotion_options:
                    # print(hand_sign)
                    # self.logger.debug(f"user is  selecting emotion index:  {self.selected_emotion_option_index}")
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
        # else:
            # self.render_ready["mediapipe"] = False
                    
    ## LLM OPTIONS CODE SECTION
    def update_selected_llm_option_mp(self, hand_sign: str):
        self.selected_llm_option_index = int(hand_sign) - 1
        for i in range(self.number_of_llm_options):
            if i == self.selected_llm_option_index:
                self.llm_options[i].hover()
            else:
                self.llm_options[i].unhover()


    def render_tts_indicator(self, canvas):
        self.tts_indicator_component.render_component(canvas)
        top_left_x, top_left_y = self.tts_indicator_component.rect_start
        bot_right_x, bot_right_y = self.tts_indicator_component.rect_end

        canvas[top_left_y: top_left_y + self.speaker_icon.shape[0], bot_right_x: bot_right_x + self.speaker_icon.shape[0]] = self.speaker_icon

    def render_tutorial_options(self,
                                canvas):
        for i in range(len(self.tutorial_options)):
            self.tutorial_options[i].update_progress()
            self.turorial_options[i].render_component(canvas)
        
            
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
        llm_options = [option.replace("’", "'") for option in llm_options]
        self.logger.info(f"ChatGPT: {llm_options}")
        print(llm_options)
        assert(len(llm_options) == self.number_of_llm_options - 2)
        self.llm_options.clear()
        self.selected_llm_option_index = -1
        
        for i in range(len(llm_options)):
            if i == 0:
                text_bottom_left = (self.frame_width//2 - 200, 200 + 100 * i)
            else:
                text_bottom_left = (
                    self.llm_options[i - 1].bot_left_list[-1][0],
                    self.llm_options[i - 1].bot_left_list[-1][1] + 2 * self.llm_options[i - 1].line_heights[-1])
            self.llm_options.append(
                OptionComponent(
                    llm_options[i],
                    line_width,
                    text_bottom_left, self.llm_progress_full_handler, show_bg=False))
        next_option_bottom_left = (
            self.llm_options[-1].bot_left_list[-1][0],
            self.llm_options[-1].bot_left_list[-1][1] + 2 * self.llm_options[-1].line_heights[-1])

        self.llm_options.append(
            OptionComponent(
                "4. NEXT OPTIONS",
                line_width,
                next_option_bottom_left,
                self.llm_next_progress_full_handler, show_bg=False))

        reset_option_bottom_left = (
            self.llm_options[-1].bot_left_list[-1][0],
            self.llm_options[-1].bot_left_list[-1][1] + 2 * self.llm_options[-1].line_heights[-1])

        self.llm_options.append(
            OptionComponent(
                "5. SIGN",
                line_width,
                reset_option_bottom_left,
                self.llm_reset_progress_full_handler, show_bg=False))

        self.render_ready["llm-options"] = True

    def llm_next_progress_full_handler(self):
        if self.render_ready['llm-options']:
            self.logger.info("DHH: NEXT OPTIONS chosen")
            self.notify_subscriber("llm-repeat-keywords")
            # self.selected_llm_option_index = -1
            self.render_ready["llm-options"] = False
        
    def llm_reset_progress_full_handler(self):
        if self.render_ready['llm-options']:
            self.logger.info("DHH: SIGN chosen")
            self.notify_subscriber("need-keyword-input")
            self.render_ready["llm-options"] = False
            if self.keyword_text:
                self.render_ready["keyword_text"] = False


    def tutorial_option_progress_full_handler(self):
        if self.render_ready['llm-options']:
            self.logger.info(f"DHH: {self.llm_options[self.selected_llm_option_index].text}")
            
            self.last_chosen_llm_option_text = copy.deepcopy(self.llm_options[self.selected_llm_option_index].text)
            self.last_chosen_llm_option_bot_left = copy.deepcopy(self.llm_options[self.selected_llm_option_index].bot_left_list[0])
            if self.emotional_voice: 
                self.render_ready["emotion_options"] = True
                self.notify_subscriber(
                    "done_choosing_llm_option", copy.deepcopy(self.llm_options[self.selected_llm_option_index].text))
            else:
                self.notify_subscriber(
                    "done_choosing_llm_option_neutral",
                    self.llm_options[self.selected_llm_option_index].text[3:]
                )
            for i in range(len(self.llm_options)):
                self.llm_options[i].reset()
            self.update_current_task_index()
            self.selected_llm_option_index = -1
            self.render_ready["llm-options"] = False

    def llm_progress_full_handler(self):
        if self.render_ready['llm-options']:
            self.logger.info(f"DHH: {self.llm_options[self.selected_llm_option_index].text}")
            self.notify_subscriber(
                "llm-add-prompt",
                "B",
                self.llm_options[self.selected_llm_option_index].text
            )
            self.last_chosen_llm_option_text = copy.deepcopy(self.llm_options[self.selected_llm_option_index].text)
            self.last_chosen_llm_option_bot_left = copy.deepcopy(self.llm_options[self.selected_llm_option_index].bot_left_list[0])
            self.render_ready["llm-options"] = False
            if self.keyword_text:
                self.render_ready["keyword_text"] = False
            if self.emotional_voice: 
                self.render_ready["emotion_options"] = True
                self.notify_subscriber(
                    "done_choosing_llm_option", copy.deepcopy(self.llm_options[self.selected_llm_option_index].text))
            else:
                self.notify_subscriber(
                    "done_choosing_llm_option_neutral",
                    self.llm_options[self.selected_llm_option_index].text[3:]
                )

            # self.selected_llm_option_index = -1

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
        self.logger.info(f"DHH sign: {text}")
        # self.keyword_text_component.reset()
        if self.keyword_text:
            self.keyword_text_component.set_text(text)
            self.render_ready["keyword_text"] = True
        
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
        self.voice_rec_text_component = TextComponent(
            text, 25, (self.canvas.shape[1] // 2 - 100, self.canvas.shape[0]))
        self.render_functions["voice_rec_text"] = lambda canvas: self.voice_rec_text_component.render_component(
            canvas)
        if self.tutorial:
            self.render_ready["llm-options"] = True
        
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
        self.logger.info(f"Speaking Start: {self.last_chosen_llm_option_text}")
        self.tts_indicator_component.set_text(self.last_chosen_llm_option_text)
        self.tts_indicator_component.set_bot_left(
            self.last_chosen_llm_option_bot_left)
        num_words = len(self.last_chosen_llm_option_text[4:].split())
        self.tts_indicator_component.set_full_progress_time(num_words * 0.5)
        self.tts_indicator_component.reset_progress()
        self.render_ready["tts_indicator"] = True

    def on_tts_end_handler(self):
        self.logger.info(f"Speaking Finished: {self.last_chosen_llm_option_text}")
        self.render_ready["tts_indicator"] = False
        self.voice_rec_text_component.set_text("Recording Voice")
        self.render_ready["voice_rec_text"] = True
        
        if self.tutorial:
            self.render_ready["llm-options"] = True


    def update_current_task_index(self):
        if self.has_tasks:
            with self.current_task_index_lock:
                if self.current_task_index == len(self.tasks) - 1:
                    return
                self.current_task_index += 1
                self.current_task_component.set_text(self.tasks[self.current_task_index])
                self.logger.info(f"Phase {self.current_task_index}:{self.tasks[self.current_task_index]}")
                

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

    
