import threading
import queue
from input_module import InputHandler
import llama_module 
from camera_module import CameraHandler

# Global variable to store the text displayed on the video feed
llama_responses_queue: queue.Queue = queue.Queue()


def update_text_on_video():
    input_handler = InputHandler()
    llama_handler = LlamaHandler()

    while True:
        print("PLEASE SPEAK NOW")
        user_input = input_handler.record_voice_input()
        print("input is : ", user_input)
        responses = llama_handler(user_input)
        llama_responses_queue.put_nowait(responses)
        # text_on_video = "\n".join(responses)


def display_output_on_video():
    camera_handler = CameraHandler()

    try:
        camera_handler.initialize_camera()
        while True:
            if llama_responses_queue.qsize() != 0:
                text_on_video = llama_responses_queue.get_nowait()
                print("text on video is", text_on_video)
                camera_handler.set_text_overlay(text_on_video)
            camera_handler.display_video_feed()
    except RuntimeError as e:
        print(str(e))
    finally:
        camera_handler.release_camera()


if __name__ == "__main__":
    # Thread for updating text
    text_thread = threading.Thread(target=update_text_on_video)
    text_thread.daemon = True
    text_thread.start()

    # Display video feed
    print("Starting camera feed. Press 'q' to exit.")
    display_output_on_video()
