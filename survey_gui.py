import logging
import numpy as np
import os
from copy import deepcopy
import cv2
import argparse
import random
import datetime
import dearpygui.dearpygui as dpg
from make_survey_clips import save_survey_clips, parse_frame_timestamp_file, get_frame_size


#################

# survey should be from 1 to 7

################
SURVEY_QUESTIONS_COMPONENT_1 = "survey_questions_component_1.txt"
SURVEY_QUESTIONS_COMPONENT_2_NORMAL = "survey_questions_component_2_normal.txt" 
SURVEY_QUESTIONS_COMPONENT_2_NEXT = "survey_questions_component_2_next.txt" 
SURVEY_QUESTIONS_COMPONENT_2_RESET = "survey_questions_component_2_reset.txt" 
SURVEY_QUESTIONS_COMPONENT_3 = "survey_questions_component_3.txt"
SURVEY_QUESTIONS_COMPONENT_4 = "survey_questions_component_4.txt"


SURVEY_QUESTIONS_LAST_QUESTION_CONDITION_1 = "survey_questions_last_question_condition_1.txt"
SURVEY_QUESTIONS_LAST_QUESTION_CONDITION_2 = "survey_questions_last_question_condition_2.txt"
SURVEY_QUESTIONS_LAST_QUESTION_CONDITION_3 = "survey_questions_last_question_condition_3.txt"
SURVEY_QUESTIONS_LAST_QUESTION_CONDITION_4 = "survey_questions_last_question_condition_3.txt"


def convert_cv_to_dpg(image, width, height):
    resize_image = cv2.resize(image, (width, height))

    data = np.flip(resize_image, 2)
    data = data.ravel()
    data = np.asfarray(data, dtype='f')

    texture_data = np.true_divide(data, 255.0)

    return texture_data

def convert_log_time_to_sec(log_timestamp: str):
    return datetime.datetime.strptime(log_timestamp, "%Y-%m-%d %H:%M:%S,%f").timestamp()


def phase_indices_log_lines(log_lines: list[str]):
    """
    from the log_lines extracted from a log file, return the indices of the
    3 phases: beginning, middle, end
    """
    
    phase_indices = []
    for i in range(len(log_lines)):
        if "Phase 0" in log_lines[i]:
            phase_indices.append(i)
        if "Phase 1" in log_lines[i]:
            phase_indices.append(i)
        if "Phase 2" in log_lines[i]:
            phase_indices.append(i)
            assert len(phase_indices) == 3
            break
    # phase_indices = [0, int(len(log_lines) / 3), int(len(log_lines) / 3 * 2)]
    return phase_indices

def extract_component_2_timestamps(event_log_filename: str, normal_number_cnt = 3):
    with open(event_log_filename, 'r') as f:
        lines = [line.rstrip() for line in f]
    component_2_lines = []
    component_2_lines_normal_lines_per_phase = []    
    phase_indices = phase_indices_log_lines(lines)
    phase_indices.append(len(lines))
    
    for i in range(1, len(phase_indices)):
        phase_lines = []
        for j in range(phase_indices[i - 1], phase_indices[i]):
            if "DHH:"  in lines[j]:
                if "RESET" in lines[j] or "NEXT" in lines[j]:
                    component_2_lines.append(j)
                else:
                    phase_lines.append(j)
        component_2_lines_normal_lines_per_phase.append(phase_lines)

    for phase_lines in component_2_lines_normal_lines_per_phase:
        # print(phase_lines)
        random_normal_lines = random.sample(phase_lines, normal_number_cnt)
        component_2_lines.extend(random_normal_lines)
    component_2_lines.sort()
    for i in range(len(lines)):
        if i in component_2_lines:
            print(lines[i])
        if i in phase_indices:
            print(lines[i])
    component_2_timestamps = [convert_log_time_to_sec(lines[index].split('-INFO')[0]) for index in component_2_lines]
    return [lines[index] for index in component_2_lines], component_2_timestamps

def frame_list_as_texture(frame_list, width, height):
    return [convert_cv_to_dpg(frame, width, height) for frame in frame_list]
    

def load_video(filename: str):
    cap = cv2.VideoCapture(filename)
    frame_list = []
    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"can't parse frame from {filename}")
            break
        frame_list.append(cv2.resize(frame, (640, 480)))
        frame_list.append(cv2.resize(frame, (640, 480)))        
    return frame_list

def get_first_frame(filename: str):
    cap = cv2.VideoCapture(filename)
    ret, frame = cap.read()
    return frame

def load_component_2_videos(component_2_clips_full_path):
    component_2_video_frames = []
    # component_2_clips_full_path = [os.path.join(component_2_clips_folder, file) for file in os.listdir(component_2_clips_folder)]
    
    for i in range(len(component_2_clips_full_path)):
        component_2_video_frames.append(load_video(component_2_clips_full_path[i]))
    return component_2_video_frames


def load_component_survey_question(filename):
    """
    return a list of questions and its type (RADIO or TEXT)
    """
    with open(filename, 'r') as f:
       questions = [tuple(line.rstrip().split('---'))for line in f]
    return questions

def get_last_questions(condition:str):
    if condition == "1": return load_component_survey_question(SURVEY_QUESTIONS_LAST_QUESTION_CONDITION_1)
    if condition == "2": return load_component_survey_question(SURVEY_QUESTIONS_LAST_QUESTION_CONDITION_2)
    if condition == "3": return load_component_survey_question(SURVEY_QUESTIONS_LAST_QUESTION_CONDITION_3)
    if condition == "4": return load_component_survey_question(SURVEY_QUESTIONS_LAST_QUESTION_CONDITION_4)

# from the index of the list, we know that first question list is component 1
# last 2 are component 3 and 4
# the rest are component 2
def get_list_survey_questions(component_2_clips_name, condition: str):
    survey_questions = []
    component_1_questions = load_component_survey_question(SURVEY_QUESTIONS_COMPONENT_1)
    component_2_questions_normal = load_component_survey_question(SURVEY_QUESTIONS_COMPONENT_2_NORMAL)
    component_2_questions_next = load_component_survey_question(SURVEY_QUESTIONS_COMPONENT_2_NEXT)
    component_2_questions_reset = load_component_survey_question(SURVEY_QUESTIONS_COMPONENT_2_RESET)
    component_3_questions = load_component_survey_question(SURVEY_QUESTIONS_COMPONENT_3)
    component_4_questions = load_component_survey_question(SURVEY_QUESTIONS_COMPONENT_4)

    
    
    # survey_questions.append(deepcopy(component_1_questions))
    
    for i in range(len(component_2_clips_name)):
        # survey_questions.append(deepcopy(component_2_questions))
        if "normal" in component_2_clips_name[i]:
            survey_questions.append(deepcopy(component_2_questions_normal))
        elif "next" in component_2_clips_name[i]:
            survey_questions.append(deepcopy(component_2_questions_next))
        elif "reset" in component_2_clips_name[i]:
            survey_questions.append(deepcopy(component_2_questions_reset))
    # survey_questions.append(deepcopy(component_3_questions))
    # survey_questions.append(deepcopy(component_4_questions))
    print("last question is",  get_last_questions(condition))
    survey_questions.append(deepcopy(get_last_questions(condition)))

    return survey_questions

class SurveyGUI():
    def __init__(
            self,
            component_2_clips_folder: str,
            survey_result_output_path: str,
            condition: str
    ):
        self.condition = condition
        self.component_2_clips_folder: str = component_2_clips_folder
        
        self.component_2_clips_name: list[str] = [filename for filename in os.listdir(self.component_2_clips_folder)]
        # print(self.component_2_clips_name)
        self.component_2_clips_name.sort(key = lambda filename: int(filename[:-4].split('_')[-1]))
        print(self.component_2_clips_name)
        
        self.component_2_clips_full_path: list[str] = [os.path.join(self.component_2_clips_folder, filename) for filename in self.component_2_clips_name]
        
        # self.component_2_frame_width, self.component_2_frame_height = get_frame_size(self.component_2_clips_full_path[0])
        self.component_2_video_window_width = 640
        self.component_2_video_window_height = 480
        self.component_2_video_frame_index = 0


        
        self.component_2_video_frames = [frame_list_as_texture(frame_list, self.component_2_video_window_width, self.component_2_video_window_height) for frame_list in load_component_2_videos(self.component_2_clips_full_path)]

        # each of the item in this questions_list is a list of questions to show on 1 screen
        self.questions_list = get_list_survey_questions(
            self.component_2_clips_name,
            self.condition)
        print(self.questions_list)
        self.current_questions_list_index = 0

        
        

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        interaction_handler = logging.FileHandler(survey_result_output_path)
        interaction_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(message)s')

        interaction_handler.setFormatter(formatter)
        self.logger.addHandler(interaction_handler)



        self.window_height = 1000
        self.window_width = 800

    def current_component(self): # assuming that self.current_questions_list_index is valid
        if self.current_questions_list_index < len(self.questions_list) - 1:
            return 2
        return -1
        # if self.current_questions_list_index == 0:
            # return 1
        # if self.current_questions_list_index == len(self.questions_list) - 2:
            # return 3
        # if self.current_questions_list_index == len(self.questions_list) - 1:
            # return 4
        # return 2

    def delete_survey_questions(self, survey_questions_container_tag: str):
        for child in dpg.get_item_children(survey_questions_container_tag)[1]:
            # print(child)
            dpg.delete_item(child)

    def load_new_survey_questions(
            self,
            survey_questions_container_tag: str,
            question_list: list[tuple[str, str]]):
        # self.delete_survey_questions(survey_questions_container_tag)
        question_cnt = 1
        print(question_list)
        for question, q_type in question_list:
            assert q_type == "RADIO" or q_type == "TEXT"
            dpg.add_text(
                question,
                label = f"Question: {question_cnt}",
                wrap = self.window_width - 50,
                parent = survey_questions_container_tag
            )
            if q_type == "TEXT":
                dpg.add_input_text(
                    label = f"INPUT Question: {question_cnt}",
                    parent = survey_questions_container_tag,
                    multiline=True)
            else:
                survey_items = ["1 (Very Poorly)", "2", "3", "4", "5", "6", "7 (Very Well)"]
                dpg.add_radio_button(
                    label = f"INPUT Question: {question_cnt}",
                    parent = survey_questions_container_tag,
                    items = survey_items,
                    default_value = "1",
                    horizontal=True
                )
            question_cnt += 1


    def save_answers_container(self, survey_questions_container_tag):
        questions_input = []
        questions = []
        print( dpg.get_item_children(survey_questions_container_tag))
        for child in dpg.get_item_children(survey_questions_container_tag)[1]:
            child_label = dpg.get_item_label(child)
            if "INPUT" in child_label:
                questions_input.append(child)
            else:
                questions.append(child)
        questions_answers = dpg.get_values(questions_input)
        questions_str = dpg.get_values(questions)
        self.logger.info(f"Component {self.current_component()}")
        if self.current_component() ==2 : 
            self.logger.info(f"Interaction {self.current_questions_list_index}")
        for i in range(len(questions)):
            self.logger.info(f"Q{i}:{questions_str[i]}:{questions_answers[i]}")
            

    def get_component_2_interaction_index(self):
        assert self.current_component() == 2
        
        # return self.current_questions_list_index - 1

        return self.current_questions_list_index


    def next_interaction_button_handler(self):
        self.save_answers_container("survey_questions")
        if self.current_questions_list_index == len(self.questions_list) - 1:
            dpg.stop_dearpygui()
            return
        

        self.delete_survey_questions("survey_questions")
        self.current_questions_list_index += 1
        if self.current_component() == 2:
            self.component_2_video_frame_index = 0
        
        self.load_new_survey_questions(
            "survey_questions",
            self.questions_list[self.current_questions_list_index])
        # return
        
        # component_2_interaction_index = self.get_component_2_interaction_index()
        # print(self.component_2_clips_name)
        

    def run(self):
        dpg.create_context()
        dpg.create_viewport(title='Survey', width=self.window_width, height=self.window_height, resizable=False)
        dpg.setup_dearpygui()
        

        black_image = np.zeros((self.component_2_video_window_width, self.component_2_video_window_height, 3))
        self.black_texture = convert_cv_to_dpg(
            black_image,
            self.component_2_video_window_width,
            self.component_2_video_window_height
        )
        # print(self.component_2_video_window_height, self.component_2_frame_width)


        with dpg.texture_registry(show=True):
            dpg.add_raw_texture(self.component_2_video_window_width,
                                self.component_2_video_window_height,
                                self.black_texture,
                                tag="texture_tag",
                                format=dpg.mvFormat_Float_rgb)
        with dpg.window(label = "Survey Window", on_close=dpg.stop_dearpygui):
            with dpg.child_window(tag="component_2_video",
                                  label="component 2 video",
                                  height = 480,
                                  width = 800, 
                                  show = True
                                  ):
                # pass
                self.component_2_video_tag = dpg.add_image("texture_tag")
            with dpg.child_window(tag="survey_questions",
                                  label="survey questions",
                                  height = 460,
                                  width = 800, 
                                  ):
                self.load_new_survey_questions("survey_questions", self.questions_list[0])
                # the default should be the first questions list
                pass
                # self.

            self.next_screen_button_tag = dpg.add_button(
                label = "Next Interaction",
                callback = self.next_interaction_button_handler)
        dpg.show_viewport(minimized=False)
        # dpg.start_dearpygui()
        
        while dpg.is_dearpygui_running():
            if self.current_component() == 2:
                # print("bruh")
                # print(self.get_component_2_interaction_index())
                dpg.set_value(
                    "texture_tag",
                    self.component_2_video_frames[self.get_component_2_interaction_index()][self.component_2_video_frame_index])
                self.component_2_video_frame_index = (self.component_2_video_frame_index + 1) % len(self.component_2_video_frames[self.get_component_2_interaction_index()])
                # print(self.component_2_video_frame_index)
            dpg.render_dearpygui_frame()

        
        dpg.destroy_context()
            # need to add callback later which changes the question list
            # and save the previous interaction to file                

    # def write_survey_res_to_file(filename, result):
        # with open(filename, 'w') as f:
            # f.write('\n'.join(result))
        



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--condition",
        help = "condition",
        choices = ["1", "2", "3", "4" , "5"]
    )
    
    parser.add_argument(
        "-lp",
        "--log-path",
    )
    
    parser.add_argument(
        "-vp",
        "--video-path",
    )
    
    parser.add_argument(
        "-rp",
        "--result-path",
    )
    parser.add_argument("-fp",
                        "--frame-path",
                        help = "path to video frame timestamp file",
                        required=True)
    parser.add_argument("-cp",
                        "--clip-path",
                        help = "output folder for clips")
    parser.add_argument("-mv",
                        "--make-video",
                        help = "make video",
                        action = "store_true")
                    
    args = parser.parse_args()
    frame_timestamp = parse_frame_timestamp_file(args.frame_path)
    
    event_lines, event_timestamp = extract_component_2_timestamps(args.log_path)
    print("event lines is", event_lines)
    def get_video_filename_list(event_lines):
        filename_list = []
        for i in range(len(event_lines)):
            if "reset" in event_lines[i].lower():
                filename_list.append(f"reset_component_2_interaction_{i}.mp4")
            elif "next" in event_lines[i].lower():
                filename_list.append(f"next_component_2_interaction_{i}.mp4")
            else:
                filename_list.append(f"normal_component_2_interaction_{i}.mp4")
        return filename_list
        

    if args.make_video:
        save_survey_clips(
            video_path=args.video_path,
            frame_timestamp = frame_timestamp,
            event_timestamp = event_timestamp,
            before_event_duration=[5 for _ in range(len(event_timestamp))],
            after_event_duration=[3 for _ in range(len(event_timestamp))],
            clip_names=get_video_filename_list(event_lines),
            output_folder=args.clip_path
        )
    survey = SurveyGUI(
        args.clip_path,
        args.result_path,
        condition=args.condition)
    # survey.setup_ui_layout()g
    survey.run()
    print(event_lines)
    
# #
# python survey_gui.py -lp ~/scifilab/realtime-llm/testing_random/interaction_log_03_20_2025_00_29_05.txt -vp /home/nam/scifilab/realtime-llm/testing_random/video/video_03_20_2025_00_29_05.mp4- fp /home/nam/scifilab/realtime-llm/testing_random/video/frame_time_video_03_20_2025_00_29_05.csv -cp /home/nam/scifilab/realtime-llm/testing_random/video/
