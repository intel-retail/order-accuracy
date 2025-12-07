import cv2
import os

class VideoReader:
    def __init__(self, source_path):
        print("[video-reader] Expected video path:", source_path)
        print("[video-reader] File exists?", os.path.exists(source_path))
        self.cap = cv2.VideoCapture(source_path)

    def frames(self):
        if not self.cap.isOpened():
            raise RuntimeError("Cannot open video. Check path and file format.")
        frame_index = 0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            yield frame_index, frame
            frame_index += 1
        self.cap.release()
