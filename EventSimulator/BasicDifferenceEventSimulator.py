"""
    Project Name: BasicDiffenceEventSimulator.py
    Description : EventSimulator (W. Basic Difference of Prev Frame&Frame) 
    Author: Youngjin Moon
    Date: 
        First Release: 23.08.14
"""
import cv2
from typing import List, Tuple
import numpy as np

class Event:
    def __init__(self, in_x, in_y, in_timestamp, in_polarity):
        self.x = in_x
        self.y = in_y
        self.timestamp = in_timestamp
        self.polarity = in_polarity

# Helper function to pack data into an event struct.
def packEvents(lighter_events, # cv::Mat containing pixel which got brighter
                darker_events, # cv::Mat containing pixel which got darker
                timestamp:int          # Time stamp of the events
                ) -> List[Event]:
    pos_polarity_events = np.transpose(np.nonzero(lighter_events))
    neg_polarity_events = np.transpose(np.nonzero(darker_events))

    output = []

    for point in neg_polarity_events:
        output.append(Event(point[1], point[0], timestamp, False))
        
    for point in pos_polarity_events:
        output.append(Event(point[1], point[0], timestamp, True))

    return output

# Implementing the simple difference frame-based event simulator.
class BasicDifferenceEventSimulator:
    def __init__(self, 
                 c_pos: int, # Positive threshold
                 c_neg: int  # Negative threshold
                 ):
        self.c_pos_ = c_pos
        self.c_neg_ = c_neg
        self.name_ = 'BasicDifferenceEventSimulator'
        self.out_frames_ = []

    # Events
    def getEvents(self,
                  prev_frame:np.array, # Frame form the previous time step
                  frame:np.array,      # Frame form the current time step
                  prev_timestamp:int, # Time stamp of the previous time step
                  timestamp:int      # Time stamp of the current time step
                #   num_frames:int          # Number of frames
                  ):
        darker = cv2.subtract(prev_frame, frame)
        lighter = cv2.subtract(frame, prev_frame)
        _, darker_binary = cv2.threshold(darker, self.c_pos_, 255, cv2.THRESH_BINARY)
        _, lighter_binary = cv2.threshold(lighter, self.c_neg_, 255, cv2.THRESH_BINARY)

        events_ = packEvents(lighter_binary, darker_binary, timestamp)

        return events_

    # Accumulated Events
    def getEventFrame(self, 
                      prev_frame,
                      frame,
                      #num_frames:int
                      ):
        # self.out_frames_ = [None] * num_frames
        darker = cv2.subtract(prev_frame, frame)
        lighter = cv2.subtract(frame, prev_frame)
       
        _, darker_binary = cv2.threshold(darker, self.c_pos_, 255, cv2.THRESH_BINARY)
        _, lighter_binary = cv2.threshold(lighter, self.c_neg_, 255, cv2.THRESH_BINARY)

        zeros = np.zeros(darker_binary.shape, dtype=np.uint8)
        channels = [lighter_binary, zeros, darker_binary]
        self.out_frames_.append(cv2.merge(channels))

        return self.out_frames_
   
    # Name of the Event Simulator
    def getName(self):
        return str(self.name_) + '_' + str(self.c_pos_) + '_' + str(self.c_neg_)
    


def main():
    pass

if __name__ == '__main__':
    main()
  
  
