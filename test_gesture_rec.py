import cv2
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


model_path = './gesture_recognizer.task'



BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode

cnt = 0
text_on_screen = ""
# Create a gesture recognizer instance with the live stream mode:
def print_result(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    print(result)
    if len(result.gestures) != 0 and result.gestures[0][0].category_name != "None":
        global text_on_screen
        text_on_screen = result.gestures[0][0].category_name
        # print('gesture recognition result: {}'.format(result))
        
options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result)

cam = cv2.VideoCapture(4)


with GestureRecognizer.create_from_options(options) as recognizer:
  # The detector is initialized. Use it here.
    while True:
        frame_time_ms = time.time_ns() // 1_000_000
        ret, frame = cam.read()
        mp_input = mp.Image(image_format=mp.ImageFormat.SRGB, data = frame)
        
        recognizer.recognize_async(mp_input, frame_time_ms)
        
        key = cv2.waitKeyEx(1)
        cv2.putText(frame, text_on_screen, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
        cv2.imshow('cam', frame)
        if key == 27:
            print("pressed esc")
            break
cam.release()
cv2.destroyAllWindows()
print(cnt)
