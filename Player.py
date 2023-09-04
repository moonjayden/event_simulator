"""
    Project Name: Player.py
    Description : Streaming the loaded Video
    Author: Youngjin Moon
    Date: 
        First Release: 23.08.14
"""
"""
    Checked List
    -(def) toGray
    -(def) getRandomHex
    -VideoPlayer <- NO

    -VideoStream <- Need
"""
# import cv2
import numpy as np
import os
import random
import time
import json
from collections import namedtuple
from Loader import *
from EventSimulator.BasicDifferenceEventSimulator import *
from EventSimulator.DenseInterpolatedEventSimulator import *

import pdb

# Color to gray converstion
def toGray(frame:cv2.Mat): # Color frame in BGR
    grey_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return grey_frame

# Returns a random hex number as string.
def getRandomHex(length:int):
    random_hex = ''
    hexChar = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f']
    for i in range(length):
        random_hex += random.choice(hexChar)
    return random_hex
"""
class VideoPlayer:
    def __init__(self, event_simulator:BasicDifferenceEventSimulator, res_x:int, res_y:int):
        self.event_simulator = event_simulator
        self.res_x = res_x
        self.res_y = res_y

        self.current_frame = 0
        # roi = (x, y, width, height)
        self.roi = None
        self.loader = OpenCVLoader()

    def simulate(self, path, height:int=0, width:int=0, repeats:int=1,
                 event_statistics:bool=False, record_video:bool=False):
        self.path = path
        self.height = height
        self.width = width
        self.repeats = repeats
        pdb.set_trace()
        self.loader.load(self.path, height, width)
        self.setupEventSimulator()
        start = cv2.getTickCount()
        self.loopSimulation(repeats, 0, event_statistics, record_video)
        end = cv2.getTickCount()
        ms = (end - start) / cv2.getTickFrequency()
        print(f"Average frame time : {ms / self.current_frame} s")
        print(f"Frames processed : {self.current_frame} frames")
        print(f"Frames per second : {self.current_frame / ms} fps")
        
        # Reset counter to be ready to play again
        self.current_frame = 0

    # Return Simulation Time
    def simulateTimed(self, path, height:int=0, width:int=0, repeats:int=1, num_frames:int=0):
        self.path = path
        self.height = height
        self.width = width
        self.repeats = repeats
        self.num_frames = num_frames
        self.loader.load(path, height, width)
        self.setupEventSimulator()
        start = cv2.getTickCount()
        self.loopSimulation(repeats, num_frames)
        end = cv2.getTickCount()
        elapsed_time = (end - start) / cv2.getTickFrequency()
        frametime = elapsed_time / (self.current_frame * repeats)
        print(f"Average frame time : {elapsed_time / self.current_frame} s")
        print(f"Frames processed : {self.current_frame} frames")
        print(f"Frames per second : {self.current_frame / elapsed_time} fps")

        # Reset counter to be ready to play again
        self.current_frame = 0
        return frametime

    # Save Single Frame
    def saveSingleFrame(self, path, height:int=0, width:int=0, frame_index:int=1):
        self.path = path
        self.height = height
        self.width = width
        self.frame_index = frame_index
        print(f'self.loader = {self.loader}')
        print(f'Num Frames = {self.loader.getNumFrames()}')
        print(f'filename = {self.loader.getFileName()}')
        print(f'frame index = {frame_index}')
        pdb.set_trace()
        if self.loader.getNumFrames() < 3 or frame_index < 3:
            print("ERROR: Frame index must be at least 3 and video must contain 3 frames")
            return
        self.loader.load(path, height, width)
        self.setupEventSimulator()
        first = toGray(self.loader.getFrame(frame_index - 2))
        second = toGray(self.loader.getFrame(frame_index - 1))
        num_frames = self.event_simulator.getEventFrame(first, second)
        results = self.event_simulator.getEventFrame(second, toGray(self.loader.getFrame(frame_index)))

        base_filename = os.path.basename(path)
        filename, _ = os.path.splitext(base_filename)
        output_path = filename
        # pdb.set_trace()
        for i, result_frame in enumerate(results):
            random_id = getRandomHex(6)
            file_name = f"{frame_index}_{self.event_simulator.getName()}.png"
            if self.roi is not None and self.roi[2] > 0 and self.roi[3] > 0:
                cv2.imwrite(os.path.join(output_path, f"{output_path}_{file_name}"), result_frame[self.roi[1]:self.roi[1] + self.roi[3], self.roi[0]:self.roi[0] + self.roi[2]])
            else:
                cv2.imwrite(os.path.join(output_path, f"{output_path}_{file_name}"), result_frame)
            print("Saved frame", frame_index, "to", os.path.join(output_path, f"{output_path}_{file_name}"))
    
    def getNextFrame(self):
        self.current_frame += 1
        if self.loader.getNumFrames() == 0:
            raise Exception("No frames in video! Did you forget to call loader->load(path)?")
        return self.loader.getFrame(self.current_frame % self.loader.getNumFrames())

    def setupEventSimulator(self):
        frame_size = self.loader.getFrame(0).shape[:2][::-1]
        event_simulator_ = self.event_simulator()
        event_simulator_.setup(frame_size)
    
    def setEventSimulator(self, event_simulator):
        self.event_simulator_ = event_simulator

    def loopSimulation(self, repeats:int=1, num_frames:int=0,
                       event_statistics:bool=False, record_video:bool=False):
        # super().load(path, height, width)
        frame_rate = self.loader.getFrameRate()
        pdb.set_trace()
        time_per_frame = 1.0 / frame_rate
        if num_frames == 0:
            num_frames = self.loader.getNumFrames()
        seconds = num_frames * time_per_frame
        frame_size = self.loader.getFrame(0).shape[:2][::-1]
        total_events_per_pixel = np.zeros(frame_size, dtype=np.float64)
        pos_events_per_pixel = np.zeros(frame_size, dtype=np.float64)
        neg_events_per_pixel = np.zeros(frame_size, dtype=np.float64)

        print("Height:", frame_size[1], ", width:", frame_size[0])
        print("Framerate:", frame_rate)
        print("Number of frames:", num_frames)
        print("Time per frame:", time_per_frame, "s")
        print("Video duration:", num_frames / frame_rate, "s")

        video_capture = None
        if record_video:
            simulator_name = self.event_simulator.getName()
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_capture = cv2.VideoWriter(f"{simulator_name}_video.mp4", fourcc, 20.0, frame_size)

        should_quit = False
        while not should_quit:
            if (repeats > 1 and repeats <= self.current_frame // num_frames) or (repeats == 1 and self.current_frame >= num_frames):
                break

            frame = toGray(self.getNextFrame())
            out_frames = self.event_simulator.getEventFrame(toGray(self.getNextFrame()), frame)

            if video_capture:
                for frame in out_frames:
                    video_capture.write(frame)

            if event_statistics:
                prev_frame = toGray(self.loader.getFrame(self.current_frame - 1))
                timestamp = self.current_frame * time_per_frame
                events = self.event_simulator.getEvents(prev_frame, frame, (self.current_frame - 1) * time_per_frame, timestamp)
                for event in events:
                    total_events_per_pixel[event.y, event.x] += 1.0
                    if event.polarity:
                        pos_events_per_pixel[event.y, event.x] += 1.0
                    else:
                        neg_events_per_pixel[event.y, event.x] += 1.0

            cv2.imshow("OpenCVPlayer", out_frames[0])
            key = cv2.waitKey(1)
            if key == 27:  # esc key
                should_quit = True
            elif key == 115:  # s key
                file_path = f"../res/export_frames/{os.path.basename(self.loader.getFileName())}_{self.current_frame}.png"
                cv2.imwrite(file_path, out_frames[0])
                print("Saved frame to file:", file_path)

        cv2.destroyAllWindows()

        if event_statistics:
            simulator_name = self.event_simulator.getName()
            np.savetxt(f"{simulator_name}_total_events_per_pixel.csv", total_events_per_pixel, delimiter=",")
            np.savetxt(f"{simulator_name}_pos_events_per_pixel.csv", pos_events_per_pixel, delimiter=",")
            np.savetxt(f"{simulator_name}_neg_events_per_pixel.csv", neg_events_per_pixel, delimiter=",")
            print("Event statistics saved.")

class OpenCVPlayer(VideoPlayer):    
    def __init__(self):
        super().__init__()

    # def load(self, path, height=0, width=0):
    #     super().load(path, height, width)
"""
class VideoStreamer():
    def __init__(self, event_simulator, Diff:bool=False):
        self.event_simulator_ = event_simulator
        self.Diff =Diff
        self.roi = None
        self.loader = OpenCVLoader()
    def simulate_from_stream(self, source_index):
        import time
        num = 0
        cap = cv2.VideoCapture(source_index)
        self.loader.load(source_index)
        if not cap.isOpened():
            raise RuntimeError("Error opening video source. Does it exist?")

        ret, prev_frame = cap.read()
        prev_frame = toGray(prev_frame)
        # self.event_simulator_.setup(prev_frame.shape)
        while True:

            ret, frame = cap.read()
            if not ret:
                break
            num_frames = self.loader.getNumFrames()
            gray_frame = toGray(frame)
            prev_frame = np.array(prev_frame)
            gray_frame = np.array(gray_frame)
            s = time.time()
            out_frames = self.event_simulator_.getEventFrame(prev_frame, gray_frame)
            e = time.time()
            print(f'Elapsed Time = {(e-s)*1000}ms')
            should_quit = False
            # pdb.set_trace()
            for i in range(1):
                cv2.imshow("OpenCVPlayer_color", frame)
                cv2.imshow("OpenCVPlayer_gray", gray_frame)
                if self.Diff:
                    if num >= 2:
                        cv2.imshow("OpenCVPlayer", out_frames[num-2])
                else:
                    cv2.imshow("OpenCVPlayer", out_frames[num])
                c = cv2.waitKey(1) & 0xFF
                if c == 27:  # esc. to quit
                    should_quit = True
                    break
            if should_quit:
                break
            prev_frame = gray_frame
            num +=1

        cap.release()
        cv2.destroyAllWindows()
    

def main():
    pass
    # video_path = "./dataset/sample.mp4"
    # a = VideoStreamer(BasicDifferenceEventSimulator(3, 3))
    # a.simulate_from_stream(video_path)
    # b = VideoStreamer(DenseInterpolatedEventSimulator(3, 3))
    # b.simulate_from_stream(video_path)

if __name__ == '__main__':
    main()
  
  



