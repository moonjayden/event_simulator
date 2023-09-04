"""
    Project Name: Loader.py
    Description : Load Video
    Author: Youngjin Moon
    Date: 
        First Release: 23.08.14
"""

import cv2
import numpy as np
import pdb


class VideoLoader:
    def __init__(self):
        self.frame_buffer = []
        self.num_frames = 0
        self.frame_rate = 0
        self.path = ""
        self.res_x = 0
        self.res_y = 0

    def getFileName(self):
        return self.path.split('/')[-1]

    def load(self, path, height=0, width=0):
        self.path = path
        self.frame_buffer = []

        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise Exception("Error reading video file. No exist")

        self.num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_rate = int(cap.get(cv2.CAP_PROP_FPS))

        if height == 0:
            self.res_y = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        else:
            self.res_y = height
        if width == 0:
            self.res_x = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        else:
            self.res_x = width

        print("Creating framebuffer of:", self.num_frames, "Frames")

        for i in range(self.num_frames):
            # ret : read true/false
            # frame : frame image, numpy array
            ret, frame = cap.read()
            if not ret:
                break

            if self.res_y > 0 and self.res_x > 0:
                resized_frame = cv2.resize(frame, (self.res_x, self.res_y), interpolation=cv2.INTER_LINEAR)
                self.frame_buffer.append(resized_frame)
            else:
                self.frame_buffer.append(frame)

        print("Finished creating framebuffer\n")

        cap.release()

    def release(self):
        self.frame_buffer = []

    def getNumFrames(self):
        return self.num_frames

    def getFrameRate(self):
        return self.frame_rate

    def getFrameHeight(self):
        return self.res_y

    def getFrameWidth(self):
        return self.res_x

    def getFrame(self, index):
        return self.frame_buffer[index]


class OpenCVLoader(VideoLoader):
    def load(self, path, height=0, width=0):
        super().load(path, height, width)
        
""""""
def main():
    loader = OpenCVLoader()
    video_path = "./dataset/sample.mp4"
    loader.load(video_path)
    # pdb.set_trace()

    print("Video File:", loader.getFileName())
    print("Number of Frames:", loader.num_frames)
    print("Frame Rate:", loader.frame_rate)
    print("Frame Height:", loader.res_y)
    print("Frame Width:", loader.res_x)

if __name__ == "__main__":
    main()
""""""