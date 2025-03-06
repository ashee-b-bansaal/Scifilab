import logging
from collections import deque, Counter
import copy
import queue
import threading
import csv
import numpy as np
import mediapipe as mp
import cv2
from keypoint_classifier.keypoint_classifier import KeyPointClassifier
import itertools
from typing import Callable
import mp_drawing_utils


with open('./keypoint_classifier/keypoint_5_digit_and_ok_classifier_label.csv',
              encoding='utf-8-sig') as f:
            keypoint_classifier_labels_f = csv.reader(f)
            keypoint_classifier_labels = [
                row[0] for row in keypoint_classifier_labels_f
            ]

def hand_sign_to_index(hand_sign: str):
    print(hand_sign)
    return 0
    
            
def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv2.boundingRect(landmark_array)

    return [x, y, x + w, y + h]



def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        # landmark_z = landmark.z

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point



def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list


class GestureRecognition:
    def __init__(self, logger: logging.Logger) -> None:
        self.keypoint_classifier = KeyPointClassifier()
        mp_hands = mp.solutions.hands
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
        )

        self.draw_bounding_rectangle = True

        self.need_response = False
        self.need_response_cond = threading.Condition()
        self.frames_q: queue.Queue = queue.Queue()

        self.event_subscribers : dict[str, dict[str, Callable]] = dict()
        self.history_len = 8
        self.prediction_history: deque = deque(maxlen=self.history_len)

    def new_frame_handler(self, frame):
        self.frames_q.put_nowait(copy.deepcopy(frame))
        with self.need_response_cond:
            self.need_response = True
            self.need_response_cond.notify_all()
            
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


    def start_recognition(self):
        while True:
            with self.need_response_cond:
                # if doesn't need response or no frame to process then sleep
                while not self.need_response or self.frames_q.empty():
                    self.need_response_cond.wait()
                frame = self.frames_q.get_nowait()
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.hands.process(frame_rgb)
                if results.multi_hand_landmarks is not None:
                    for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                        # Bounding box calculation
                        brect = calc_bounding_rect(frame, hand_landmarks)
                        # Landmark calculation
                        landmark_list = calc_landmark_list(frame, hand_landmarks)
                        # Conversion to relative coordinates / normalized coordinates
                        pre_processed_landmark_list = pre_process_landmark(
                            landmark_list)

                        # Hand sign classification
                        hand_sign_id = self.keypoint_classifier(pre_processed_landmark_list)

                        hand_sign = keypoint_classifier_labels[hand_sign_id]
                        self.prediction_history.append(hand_sign)
                        most_common_pred = Counter(
                                    self.prediction_history).most_common()[0][0]
                        self.notify_event_subscriber(
                            "finished_processing_frame",
                            "gui",
                            self.draw_bounding_rectangle, brect, most_common_pred, landmark_list
                        )
                        self.need_response = False

if __name__ == "__main__":
    cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cam.set(cv2.CAP_PROP_FPS, 30.0)
    cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    gesture_rec = GestureRecognition()
    gesture_rec_thread = threading.Thread(
        target = gesture_rec.start_recognition,
        daemon=True
    )
    gesture_rec_thread.start()
    canvas = np.zeros((480, 640, 3), dtype=np.uint8)
    render_functions = dict()
    render_ready = dict()

    
    def render_mediapipe(draw_bounding_rect, brect, hand_sign, landmark_list):
        render_functions["mp"] = lambda: mp_drawing_utils.draw_gesture_and_landmarks_on_image(
            canvas,
            draw_bounding_rect,
            brect,
            hand_sign,
            landmark_list)
        print(hand_sign)
        render_ready["mp"] = True
        
    
    gesture_rec.register_event_subscriber(
        "finished_processing_frame",
        "gui",
        render_mediapipe)

    
    while True:
        ret, frame = cam.read()
        if not ret:
            break
        key = cv2.waitKeyEx(1) & 0xFF
        if key == 27:
            break
        
        canvas = frame
        gesture_rec.new_frame_handler(canvas)
        for key in render_ready:
            if render_ready[key]:
                render_functions[key]()
        cv2.imshow("testing gesture recognition", canvas)
    cam.release()
    
        
    


                
        
        
        
