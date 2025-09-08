import random
from switch_tasks_gui import SwitchTaskGUI
from functools import partial
from datetime import datetime
import argparse
from voice_recognition import VoiceRecognition
from keyboard_gui import KeyboardGUIInput
from kbd_input import KeyboardInput
from video_recorder import VideoRecorder
from chatgpt_api import ChatGPTAPI
from android_input import AndroidInput
from tcp_server import TCPServer, IP, PORT
from tts import TTS, Emotions
from events import TCPServerEvents, AndroidInputEvents
from gesture_recognition import GestureRecognition
from gui import GUIClass, QueueHandler, delete_contents_folder, flatten, TextComponent, OptionComponent
import logging
import os
import threading
from typing import Union
from camera_module import OpenCVCamera

def parse_task_file(filename: str):
    with open(filename, 'r') as f:
        lines = ("\n".join([line.rstrip() for line in f])).split("\n---\n")
    print(lines)
    return lines


def setup_data_collection_mode(
        folder_path: str,
        task_path: str,
        fullscreen: bool,
        black: bool,
        emotional_voice: bool,
        voice_gender: str,
        output_index: Union[int, None],
        input_index: Union[int, None],
        word_input: str,
        condition: str,
        image_context:bool,
        logger: logging.Logger,
):
    ### initialize the files and folders
    video_path = os.path.join(folder_path, "video")
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
    hand_cam = OpenCVCamera(6, 720, 1280, 30.0)
    interaction_cam = OpenCVCamera(4, 720, 1280, 30.0)
    # tasks = ["bruh"]
    tasks = []
    if task_path != "":
        tasks = parse_task_file(task_path)
    gui: GUIClass = GUIClass(exit_event=exit_event,
                             fullscreen=fullscreen,
                             black=black,
                             hand_cam = hand_cam,
                             interaction_cam=interaction_cam,
                             logger=logger,
                             emotional_voice=emotional_voice,
                             tasks=tasks,
                             tutorial=False,
                             image_context=image_context
                             )
    video_recorder = VideoRecorder(video_path, video_filename, frame_size=(gui.frame_width, gui.frame_height))

    voice_rec: VoiceRecognition = VoiceRecognition(
        input_device_index=input_index,
        logger=logger,
        on_voice_recording_start=gui.on_voice_recording_start
    )

    def tts_finished_speaking_handler():
        print(gui.llm_options[0].bot_left_list)
        voice_rec.voice_start_handler()
        gui.on_tts_end_handler()
        

    tts: TTS = TTS(
        tts_start_handler=gui.on_tts_start_handler,
        tts_end_handler=tts_finished_speaking_handler,
        gender=voice_gender,
        output_device_index=output_index,
        logger=logger)

    gesture_rec: GestureRecognition = GestureRecognition(
        logger=logger)

    
    llm: ChatGPTAPI = ChatGPTAPI(
        logger=logger,
    )
    
    if word_input == "keyboard":
        keyboard_input = KeyboardGUIInput(
            new_keyboard_input_handler_list=[
                llm.keyword_input_handler, gui.new_keyword_handler],
            logger=logger,
            switch_tasks=True,
            on_button_click=gui.update_current_task_index
            
        )
        

        # keyboard_input.register_event_subscriber("keyboard_input_ready",
                                                 # "llm",
                                                 # llm.keyword_input_handler)

                
        keyboard_input_log_handler = QueueHandler(
            keyboard_input.new_message_handler)
        
        formatter = logging.Formatter('%(message)s\n')
        keyboard_input_log_handler.setFormatter(formatter)
        keyboard_input_log_handler.setLevel(logging.INFO)
        logger.addHandler(keyboard_input_log_handler)
        
        # voice_rec.register_event_subscriber(
            # "voice_input_ready",
            # "keyword_input",
            # keyboard_input.need_keyboard_input)

        gui.register_subscriber("need-keyword-input",
                                keyboard_input.need_keyboard_input)

        keyboard_input_thread: threading.Thread = threading.Thread(
            target=keyboard_input.run,
            daemon=True
        )

        keyboard_input_thread.start()
    elif word_input == "android":
        pass
        # tcp_serv = TCPServer(logger=logger)
        # android_input = AndroidInput(logger=logger)
        # tcp_serv.register_event_subscriber(
        #     TCPServerEvents.MSG_RECEIVED,
        #     "android_input",
        #     android_input.server_msg_received_handler
        # )

        # tcp_serv_thread = threading.Thread(target=tcp_serv.start_server, daemon=True)
        # tcp_serv_thread.start()
        # android_input.register_event_subscriber(AndroidInputEvents.INPUT_READY,
        #                                         "llm",
        #                                         llm.keyword_input_handler)
        # android_input.register_event_subscriber(AndroidInputEvents.INPUT_READY,
        #                                         "gui",
        #                                         gui.new_keyword_handler)
        # voice_rec.register_event_subscriber("voice_input_ready",
        #                                     "keyword_input",
        #                                     android_input.need_input_handler)
        # android_input_thread: threading.Thread = threading.Thread(
        #     target=android_input.start_input,
        #     daemon=True
        # )
        # android_input_thread.start()
        # gui.register_subscriber("need-keyword-input",
        #                         android_input.need_input_handler)
        
        # tcp_server_handler = QueueHandler(tcp_serv.new_msg_to_write_handler)
        # formatter = logging.Formatter('%(message)s\n')
        # tcp_server_handler.setFormatter(formatter)
        # tcp_server_handler.setLevel(logging.INFO)
        # logger.addHandler(tcp_server_handler)
        
    if gui.emotional_voice:
        gui.register_subscriber("done_choosing_llm_option", tts.add_text_handler)
        gui.register_subscriber(
            "done_choosing_emotion_option",
            tts.add_emotion_handler
        )
    else:
        gui.register_subscriber("done_choosing_llm_option_neutral", partial(tts.add_tts_handler, Emotions.NEUTRAL))


    gui.register_subscriber("voice-start", voice_rec.voice_start_handler)
    gui.register_subscriber("llm-add-prompt", llm.add_prompt_handler)
    gui.register_subscriber("new-frame-to-record",
                            video_recorder.new_frame_event_handler)
    gui.register_subscriber("new-frame-to-process",
                            gesture_rec.new_frame_handler)
    gui.register_subscriber("video-record-exit",
                           video_recorder.exit_event_handler)
    gui.register_subscriber("new-image-context",
                            llm.add_new_context_image)

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
    if gui.keyword_text:
        gui.add_ui_component("keyword_text", gui.keyword_text_component.render_component)
    gui.add_ui_component("voice_recording_start", lambda x: x)
    gui.add_ui_component("emotion_options", gui.render_emotion_options)
    gui.add_ui_component("mediapipe", lambda: None)
    gui.add_ui_component("tts_indicator", gui.render_tts_indicator)
    if gui.has_tasks:
        gui.add_ui_component("current_task", gui.current_task_component.render_component)

    voice_rec_thread: threading.Thread = threading.Thread(
        target=voice_rec.start_voice_input,
        daemon=True
    )
    llm_thread: threading.Thread = threading.Thread(
        target=llm.start_conversation,
        daemon=True
    )
    llm_image_thread: threading.Thread = threading.Thread(
        target=llm.context_image_start,
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

    logger.debug(args)
    logger.info(f"Condition: {condition}")

    gesture_rec_thread.start()
    voice_rec_thread.start()
    llm_thread.start()
    llm_image_thread.start()
    video_recorder_thread.start()
    tts_thread.start()
    gui.render()
    
    video_recorder_thread.join()


def generate_tutorial_tasks(num_trials: int):
    num_trials_per_digits: int = num_trials // 5
    print("num_trials_per_digits", num_trials_per_digits)
    tasks = []
    for digit in range(1, 6):
        for i in range(num_trials_per_digits):
            
            tasks.append(f"Please Choose {digit}")

    random.shuffle(tasks)
    print("number of tasks is", len(tasks))
    for i in range(len(tasks)):
        tasks[i] = tasks[i] +  f" ({i + 1}/{num_trials})."
    tasks.append("Finished Tutorial, Please Notify Researcher")
    
    print("gen tasks",tasks)
    return tasks
    

def setup_tutorial_mode(
        folder_path: str,
        fullscreen: bool,
        black: bool,
        emotional_voice: bool,
        voice_gender: str,
        output_index: Union[int, None],
        input_index: Union[int, None],
        num_trials: int,
        logger: logging.Logger,
):
    video_path = os.path.join(folder_path, "video")
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
    hand_cam = OpenCVCamera(6, 720, 1280, 30.0)
    interaction_cam = OpenCVCamera(4, 720, 1280, 30.0)
    tasks = generate_tutorial_tasks(num_trials)
    gui: GUIClass = GUIClass(exit_event=exit_event,
                             fullscreen=fullscreen,
                             black=black,
                             hand_cam = hand_cam,
                             interaction_cam=interaction_cam,
                             logger=logger,
                             emotional_voice=emotional_voice,
                             tasks=tasks,
                             tutorial=True
                             )
    video_recorder = VideoRecorder(video_path, video_filename, frame_size=(gui.frame_width, gui.frame_height))

    def tts_finished_speaking_handler():
        print(gui.llm_options[0].bot_left_list)
        gui.on_tts_end_handler()

    tts: TTS = TTS(
        tts_start_handler=gui.on_tts_start_handler,
        tts_end_handler=tts_finished_speaking_handler,
        gender=voice_gender,
        output_device_index=output_index,
        logger=logger)

    gesture_rec: GestureRecognition = GestureRecognition(
        logger=logger)

            
    if gui.emotional_voice:
        gui.register_subscriber("done_choosing_llm_option", tts.add_text_handler)
        gui.register_subscriber(
            "done_choosing_emotion_option",
            tts.add_emotion_handler
        )
    else:
        gui.register_subscriber("done_choosing_llm_option_neutral", partial(tts.add_tts_handler, Emotions.NEUTRAL))

    gui.register_subscriber("new-frame-to-process",
                            gesture_rec.new_frame_handler)
    gui.register_subscriber("new-frame-to-record",
                            video_recorder.new_frame_event_handler)
    gui.register_subscriber("video-record-exit",
                           video_recorder.exit_event_handler)
        
    gesture_rec.register_event_subscriber("finished_processing_frame",
                                          "gui",
                                          gui.mediapipe_callback_handler)

    for i in range(0, 5):
        if i == 0:
            text_bottom_left = (gui.frame_width//2 - 200, 200 + 100 * i)
        else:
            text_bottom_left = (
                gui.llm_options[i - 1].bot_left_list[-1][0],
                gui.llm_options[i - 1].bot_left_list[-1][1] + 2 * gui.llm_options[i - 1].line_heights[-1])
        
        tutorial_option = OptionComponent(
            f"{i + 1}. Choose Option {i + 1}",
            36,
            text_bottom_left,
            gui.tutorial_option_progress_full_handler)
        gui.llm_options.append(tutorial_option)


    gui.add_ui_component("llm-options", gui.render_llm_options)
    if gui.keyword_text:
        gui.add_ui_component("keyword_text", gui.keyword_text_component.render_component)
    gui.add_ui_component("emotion_options", gui.render_emotion_options)
    gui.add_ui_component("mediapipe", lambda: None)
    gui.add_ui_component("tts_indicator", gui.render_tts_indicator)
    if gui.has_tasks:
        gui.add_ui_component("current_task", gui.current_task_component.render_component)
        

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

    logger.debug(args)
    gesture_rec_thread.start()
    video_recorder_thread.start()
    tts_thread.start()
    gui.render()
    
    video_recorder_thread.join()


    
if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-p",
        "--path",
        help="path to output the data like video, voice recording,...",
        required=True)
    parser.add_argument(
        "-tp",
        "--task-path",
        help="path to output the task.txt file contaiing the differernt tasks",
        default = ""
    )
    parser.add_argument(
        "-ti",
        "--task-input",
        help="whether to use the laptop gui or android phone to switch tasks",
        choices = ["laptop", "android"],
        default = 'laptop',
        nargs = "?",
        const = "laptop"
    )
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
        "--output-index",
        help="audio output index, based on python -m sounddevice",
        type=int,
    )
    parser.add_argument(
        "-ii",
        "--input-index",
        help="audio input index, based on list_audio_indices.py",
        type=int,
    )
    parser.add_argument(
        "-c",
        "--condition",
        help = "select condition",
        choices = ["1", "2", "3", "4", "5"],
        required = True
    )
    parser.add_argument(
        "-em",
        "--emotional-voice",
        help = "whether to use emotional voice or not",
        action = "store_true"
    )
    parser.add_argument(
        "-t",
        "--tutorial",
        help = "tutorial mode",
        action = "store_true"
    )
    parser.add_argument(
        "-nt",
        "--num-trials",
        help = "tutorial mode",
        type=int
    )
    parser.add_argument(
        "-ic",
        "--image-context",
        help = "whether to use the images captured with the interaction cam for context",
        action="store_true"
    )

    args = parser.parse_args()

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

    if args.tutorial:
        setup_tutorial_mode(
            folder_path=args.path,
            fullscreen=args.fullscreen,
            black=args.black,
            emotional_voice=args.emotional_voice,
            voice_gender=args.voice_gender,
            output_index=args.output_index,
            input_index=args.input_index,
            num_trials=args.num_trials,
            logger=logger)

    else:
        setup_data_collection_mode(folder_path=args.path,
                                   task_path=args.task_path,
                                   fullscreen=args.fullscreen,
                                   black=args.black,
                                   emotional_voice=args.emotional_voice,
                                   voice_gender=args.voice_gender,
                                   output_index=args.output_index,
                                   input_index=args.input_index,
                                   word_input=args.word_input,
                                   condition=args.condition,
                                   image_context=args.image_context,

                                   logger=logger)
        
        


    
