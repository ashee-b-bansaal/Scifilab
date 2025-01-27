import cv2

class CameraHandler():
    def __init__(self):
        self.cap = None
        self.text_overlay = ""

    def initialize_camera(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Error: Unable to access the camera.")

    def release_camera(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def set_text_overlay(self, text):
        self.text_overlay = text

    def display_video_feed(self):
        if self.cap is None:
            raise RuntimeError("Camera is not initialized.")

        while True:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                print("Error: Unable to capture video frame.")
                break

            # Add text overlay to the video frame
            font = cv2.FONT_HERSHEY_SIMPLEX
            position = (50, 50)
            font_scale = 1
            font_color = (0, 255, 0)
            thickness = 10

            cv2.putText(frame, self.text_overlay, position, font, font_scale, font_color, thickness)
            cv2.imshow('Video Feed', frame)

            # Exit on 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.release_camera()
        cv2.destroyAllWindows()
