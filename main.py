import threading
from input_module import InputHandler
from llama_module import LlamaHandler
from camera_module import CameraHandler

# Global variable to store the text displayed on the video feed
text_on_video = ""

def update_text_on_video():
    global text_on_video
    input_handler = InputHandler()
    llama_handler = LlamaHandler()

    while True:
        user_input = input_handler.record_voice_input()
        responses = llama_handler.answer(user_input)
        text_on_video = "\n".join(responses)


def display_output_on_video():
    global text_on_video
    camera_handler = CameraHandler()

    try:
        camera_handler.initialize_camera()
        while True:
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
