from datetime import datetime
import os
import shutil
import argparse
import copy
import time
import cv2
import numpy as np
import threading
import math
from typing import Callable, List, Mapping, Optional, Tuple, Union
from voice_recognition import VoiceRecognition
import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import textwrap
from kbd_input import KeyboardInput
from video_recorder import VideoRecorder
from chatgpt_api import ChatGPTAPI
from android_input import AndroidInput
from tcp_server import TCPServer
from tts import TTS, Emotions
from events import *



# recorder = AudioToTextRexcorder(
#     language="en",
#     # on_recording_start=gui.on_voice_recording_start,
#     # on_recording_stop=gui.on_voice_recording_stop
# )

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54)  # vibrant green


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
        self.text_list = textwrap.wrap(text, line_width)

    def render_component(self, canvas):
        y_offset = 0
        for line in self.text_list:
            cv2.putText(canvas, line, (self.bot_left[0], self.bot_left[1] + y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3, cv2.LINE_AA)
            y_offset += 30


class OptionComponent():
    def __init__(self,
                 text: str,
                 line_width: int,
                 bot_left: Tuple[int, int],
                 progress_100_callback: Callable,
                 progress_speed=1):
        """
        bot_left means the bottom left of where the intended string is
        supposed to go, which is the first line of the wrapped string
        """
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
        self.original_selection_time = 0

        self.register_event_subscriber(
            "progress_100", "gui", progress_100_callback)

    def render_component(self, canvas):
        if self.is_selected:
            self.bold()
        else:
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


def hand_highest_point(detection_result, image_cols, image_rows):
    highest_pixel_pos = 100000
    hand_landmarks_list = detection_result.hand_landmarks

    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
        ])

        landmark_px_list = [_normalized_to_pixel_coordinates(
            landmark.x, landmark.y, image_cols, image_rows) for landmark in hand_landmarks_proto.landmark]
        for px in landmark_px_list:
            if px is not None:
                w, h = px
                highest_pixel_pos = min(highest_pixel_pos, h)
    if highest_pixel_pos == 100000:
        return None
    return highest_pixel_pos


def draw_landmarks_on_image(rgb_image, detection_result):
    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness

    # Loop through the detected hands to visualize.
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]

        # Draw the hand landmarks.
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
        ])
        solutions.drawing_utils.draw_landmarks(
            rgb_image,
            hand_landmarks_proto,
            solutions.hands.HAND_CONNECTIONS,
            solutions.drawing_styles.get_default_hand_landmarks_style(),
            solutions.drawing_styles.get_default_hand_connections_style())
        # Get the top left corner of the detected hand's bounding box.
        hand_high = hand_highest_point(
            detection_result, rgb_image.shape[1], rgb_image.shape[0])

        cv2.line(rgb_image, (0, hand_high),
                 (1300, hand_high), (0, 0, 255), thickness=2)


class GUIClass():
    """
    The GUI class is responsible for displaying the GUI.

    As this is simulating an AR app, all gui objects will be drawn
    onto the frame recieved by the camera.

    """

    def __init__(self, exit_event: threading.Event, use_mediapipe=True, fullscreen=False, black=False):
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
        self.canvas = np.zeros(0)

        self.frame_height = -1
        self.frame_width = -1

        # if exit_event is set then must end.
        self.exit_event = exit_event

        # more robust is a State Machine to transition between states
        # right now, the render_order and render_ready are used to determine
        # what the current state is

        # variables for llm rendering
        self.llm_options: list[OptionComponent] = []
        self.selected_llm_option_index = 0
        self.number_of_llm_options = 3

        self.use_mediapipe = use_mediapipe
        if self.use_mediapipe:
            self.add_ui_component("mediapipe", lambda: None)
        self.options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
            running_mode=VisionRunningMode.LIVE_STREAM,
            result_callback=self.mediapipe_callback_handler)

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

    def update_canvas(self):
        """
        draws everything on canvas
        """
        for ui_element in self.render_order:
            if self.render_ready[ui_element]:
                self.render_functions[ui_element]()

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
        cam = cv2.VideoCapture(4)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cam.set(cv2.CAP_PROP_FPS, 30.0)
        cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(self.frame_width)
        with HandLandmarker.create_from_options(self.options) as landmarker:
            while True:
                ret, self.canvas = cam.read()
                if not ret:
                    print("can't parse frame")
                    break
                if self.use_mediapipe:
                    mp_input = mp.Image(
                        image_format=mp.ImageFormat.SRGB,
                        data=self.canvas
                    )
                    landmarker.detect_async(
                        mp_input,
                        time.time_ns() // 1_000_000
                    )
                if self.black:
                    self.canvas = np.zeros(self.canvas.shape)
                self.update_canvas()
                self.notify_subscriber(
                    "new-frame-to-record", copy.deepcopy(self.canvas))
                self.handle_input()
                if self.fullscreen:
                    cv2.namedWindow("realtime llm", cv2.WND_PROP_FULLSCREEN)
                    cv2.setWindowProperty(
                        "realtime llm", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                else:
                    cv2.namedWindow("realtime llm", cv2.WINDOW_GUI_NORMAL)
                cv2.imshow("realtime llm", cv2.resize(
                    self.canvas, (1920, 1080), interpolation=cv2.INTER_CUBIC))
                if self.exit_event.is_set():
                    break

    def mediapipe_callback_handler(self, result, output_image, timestamp_ms):
        self.render_functions["mediapipe"] = lambda: draw_landmarks_on_image(
            self.canvas,
            result)
        self.render_ready["mediapipe"] = True
        self.update_selected_option_mp(highest_point=hand_highest_point(
            result, self.canvas.shape[1], self.canvas.shape[0]))

    def update_selected_option_mp(self, highest_point):
        i = 0
        if highest_point is None:
            return
        for option in self.llm_options:
            top_left, bot_right = option.top_left, option.bot_right
            if highest_point >= top_left[1] and highest_point <= bot_right[1]:
                self.selected_llm_option_index = i
                return
            i += 1

    def render_llm_options(self):
        for i in range(len(self.llm_options)):
            if i == self.selected_llm_option_index:
                self.llm_options[i].is_selected = True
                self.llm_options[i].color = (0, 0, 255)
            else:
                self.llm_options[i].is_selected = False
                self.llm_options[i].color = (0, 255, 0)

            self.llm_options[i].update_progress()
            self.llm_options[i].render_component(self.canvas)

    def llm_response_handler(self, response: str):
        # ui_component_name = "llm-options"
        # responses_list = response.splitlines()
        # responses_list = [rep for rep in responses_list if rep[0].isdigit()]
        # def render_options():
        # self.render_functions[ui_component_name]
        print("response recieved by gui thread")

        line_width = 36
        llm_options = [rep for rep in response.splitlines() if len(
            rep) > 2 and rep[0].isdigit()]
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
                    text_bottom_left, self.progress_full_handler))

        self.render_functions["llm-options"] = lambda: self.render_llm_options()
        self.render_ready["llm-options"] = True

        self.render_ready["voice_rec_text"] = False

    def progress_full_handler(self):
        if self.render_ready['llm-options']:
            self.notify_subscriber(
                "llm-add-prompt",
                "B",
                self.llm_options[self.selected_llm_option_index].text
            )
            print("speaking, ", copy.deepcopy(
                self.llm_options[self.selected_llm_option_index].text))
            self.notify_subscriber("need_tts",Emotions.NEUTRAL,copy.deepcopy(
                self.llm_options[self.selected_llm_option_index].text))
            self.render_ready["llm-options"] = False

    def voice_input_ready_handler(self, text: str):
        self.render_ready["voice_rec_text"] = True
        self.voice_rec_text_component = VoiceRecTextComponent(
            text, 36, (self.canvas.shape[1] // 2 - 100, self.canvas.shape[0] // 2 - 100))
        self.render_functions["voice_rec_text"] = lambda: self.voice_rec_text_component.render_component(
            self.canvas)

    def on_voice_recording_start(self):
        """
        Shows Recording Voice on screen to indicate that the hearing person's voice has been
        picked up
        """
        self.render_ready["voice-recording-start"] = True
        self.render_functions["voice-recording-start"] = lambda: cv2.putText(
            self.canvas,
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




if __name__ == "__main__":
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
        default = 'keyboard',
        nargs = "?",
        const = "keyboard"
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

    args = parser.parse_args()

    folder_path = ""
    video_path = ""

    if os.path.exists(args.path):
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
        os.mkdir(args.path)
        os.mkdir(os.path.join(args.path, "video"))

    folder_path = args.path
    video_path = os.path.join(args.path, "video")
    time_rn = datetime.now().strftime("%H_%M_%S")
    video_filename = f"video_{time_rn}.mp4"

    video_recorder = VideoRecorder(video_path, video_filename)

    exit_event: threading.Event = threading.Event()
    gui: GUIClass = GUIClass(exit_event=exit_event,
                             use_mediapipe=True,
                             fullscreen=args.fullscreen,
                             black=args.black)
    voice_rec: VoiceRecognition = VoiceRecognition()

    llm: ChatGPTAPI = ChatGPTAPI()
    tts: TTS = TTS(finished_speaking_handler=voice_rec.voice_start_handler, gender=args.voice_gender)
    
    
    if args.word_input == "keyboard":
        keyboard_input = KeyboardInput()

        keyboard_input.register_event_subscriber("keyboard_input_ready",
                                             "llm",
                                             llm.keyword_input_handler)
        voice_rec.register_event_subscriber("voice_input_ready",
                                        "keyword_input",
                                        keyboard_input.voice_input_ready_handler)

        keyboard_input_thread: threading.Thread = threading.Thread(
            target=keyboard_input.start_input,
            daemon=True
        )
        keyboard_input_thread.start()
    elif args.word_input == "android":
        tcp_serv = TCPServer()
        android_input = AndroidInput()
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
        voice_rec.register_event_subscriber("voice_input_ready",
                                            "keyword_input",
                                            android_input.voice_input_ready_handler)
        android_input_thread: threading.Thread = threading.Thread(
            target=android_input.start_input,
            daemon=True
        )
        android_input_thread.start()

    gui.register_subscriber("need_tts", tts.add_tts_handler)
    gui.register_subscriber("voice-start", voice_rec.voice_start_handler)
    gui.register_subscriber("llm-add-prompt", llm.add_prompt_handler)
    gui.register_subscriber("new-frame-to-record",
                            video_recorder.new_frame_event_handler)
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

    gui.add_ui_component("llm-options", lambda x: x,)
    gui.add_ui_component("voice_rec_text", lambda x: x)
    gui.add_ui_component("voice-recording-start", lambda x: x)

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

    voice_rec_thread.start()
    llm_thread.start()
    video_recorder_thread.start()
    tts_thread.start()
    gui.render()
    
    video_recorder_thread.join()
