"""
    Project Name: DifferenceInterpolatedEventSimulator.py
    Description : 
    Author: Youngjin Moon
    Date: 
        First Release: 23.08.21
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
    # pdb.set_trace()

    output = []

    for point in neg_polarity_events:
        output.append(Event(point[1], point[0], timestamp, False))
        
    for point in pos_polarity_events:
        output.append(Event(point[1], point[0], timestamp, True))

    return output


# Implementing Difference interpolated-based event simulator.
class DifferenceInterpolatedEventSimulator:
    def __init__(self, 
                 optical_flow_calculator, 
                 num_inter_frames, 
                 c_pos, # Positive threshold
                 c_neg,  # Negative threshold
                 c_pos_inter, # Positive threshold
                 c_neg_inter  # Negative threshold
                 ):
        self.num_inter_frames = num_inter_frames
        self.optical_flow_calculator = optical_flow_calculator
        self.name = "DifferenceInterpolatedEventSimulator"
        self.c_pos = c_pos
        self.c_neg = c_neg
        self.c_pos_inter_ = c_pos_inter
        self.c_neg_inter_ = c_neg_inter
        self.events_ = []
        self.out_frames_ = []
        self.i = 0

    def getEvents(self,
                  prev_frame:np.array, # Frame form the previous time step
                  frame:np.array,      # Frame form the current time step
                  prev_timestamp:int, # Time stamp of the previous time step
                  timestamp:int      # Time stamp of the current time step
                  ):
        size = prev_frame.shape
        cols = size[1]
        rows = size[0]
        type = prev_frame.dtype
        
        darker = cv2.subtract(prev_frame, frame)
        _, darker_mask = cv2.threshold(darker, self.c_pos_inter_, 255, cv2.THRESH_TOZERO)
        lighter = cv2.subtract(frame, prev_frame)
        _, lighter_mask = cv2.threshold(lighter, self.c_neg_inter_, 255, cv2.THRESH_TOZERO)
        
        # _, darker_events = cv2.threshold(darker, self.c_pos, 255, cv2.THRESH_BINARY)
        # _, lighter_events = cv2.threshold(lighter, self.c_neg, 255, cv2.THRESH_BINARY)

        self.darker_cached_mask = np.zeros_like(darker_mask, dtype=np.uint8)
        self.lighter_cached_mask = np.zeros_like(lighter_mask, dtype=np.uint8)

        self.events_.clear()

        prev_darker_points = np.transpose(np.nonzero(self.darker_cached_mask))
        prev_lighter_points = np.transpose(np.nonzero(self.lighter_cached_mask))

        next_darker_points = self.optical_flow_calculator.calculateFlow(self.darker_cached_mask, darker_mask, prev_darker_points)
        next_lighter_points = self.optical_flow_calculator.calculateFlow(self.lighter_cached_mask, lighter_mask, prev_lighter_points)
        
        x_res, y_res = darker_mask.shape

        for j in range(self.num_inter_frames):
            alpha = (j + 1) / (self.num_inter_frames + 1)

            for i in range(len(next_lighter_points)):
                inter_point = prev_lighter_points[i] + (next_lighter_points[i] - prev_lighter_points[i]) * alpha
                current_timestamp = prev_timestamp + (timestamp - prev_timestamp) * alpha
                self.events_.append(Event(inter_point[0], inter_point[1], current_timestamp, True))

            for i in range(len(next_darker_points)):
                inter_point = prev_darker_points[i] + (next_darker_points[i] - prev_darker_points[i]) * alpha
                current_timestamp = prev_timestamp + (timestamp - prev_timestamp) * alpha
                self.events_.append(Event(inter_point[0], inter_point[1], current_timestamp, False))

        self.darker_cached_mask = darker_mask
        self.lighter_cached_mask = lighter_mask

        return self.events_

    # Accumulated Events
    def getEventFrame(self, 
                      prev_frame,
                      frame,
                      ):
        size = prev_frame.shape
        cols = size[1]
        rows = size[0]
        type = prev_frame.dtype
        darker = cv2.subtract(prev_frame, frame)
        _, darker_mask = cv2.threshold(darker, self.c_pos_inter_, 255, cv2.THRESH_TOZERO)
        lighter = cv2.subtract(frame, prev_frame)
        _, lighter_mask = cv2.threshold(lighter, self.c_neg_inter_, 255, cv2.THRESH_TOZERO)
        # Option to also add events from the subtract of the current frame and the previous frame
        _, darker_events = cv2.threshold(darker, self.c_pos, 255, cv2.THRESH_BINARY)
        _, lighter_events = cv2.threshold(lighter, self.c_neg, 255, cv2.THRESH_BINARY)
        if self.i == 0 :
            self.darker_cached_mask = darker_mask
            self.lighter_cached_mask = lighter_mask
            self.i +=1
        else:
            prev_darker_points = np.transpose(np.nonzero(self.darker_cached_mask))
            prev_lighter_points = np.transpose(np.nonzero(self.lighter_cached_mask))

            next_darker_points = self.optical_flow_calculator.calculateFlow(self.darker_cached_mask, darker_mask, prev_darker_points)
            next_lighter_points = self.optical_flow_calculator.calculateFlow(self.lighter_cached_mask, lighter_mask, prev_lighter_points)

            time = np.zeros_like(lighter_mask, dtype=np.uint8)
            for j in range(self.num_inter_frames):
                alpha = (j + 1) / (self.num_inter_frames + 1)
                # pdb.set_trace()
                for i in range(len(next_lighter_points)):
                    inter_point = prev_lighter_points[i] + (next_lighter_points[i] - prev_lighter_points[i]) * alpha
                    lighter_events[min(1279,int(inter_point[0])), min(719,int(inter_point[1]))] = 255

                for k in range(len(next_darker_points)):
                    inter_point = prev_darker_points[k] + (next_darker_points[k] - prev_darker_points[k]) * alpha
                    darker_events[min(1279,int(inter_point[0])), min(719,int(inter_point[1]))] = 255

            self.darker_cached_mask = darker_mask
            self.lighter_cached_mask = lighter_mask
            self.out_frames_.append(cv2.merge([lighter_events, time, darker_events]))

        return self.out_frames_
        

    # Name of th Event Simulator
    def getName(self):
        return str(self.name_) + '_' + str(self.c_pos_) + '_' + str(self.c_neg_)
    


def main():
    pass

if __name__ == '__main__':
    main()