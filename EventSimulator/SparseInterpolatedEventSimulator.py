"""
    Project Name: SparseInterpolatedEventSimulator.py
    Description : 
    Author: Youngjin Moon
    Date: 
        First Release: 23.08.18
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

# Implementing the simple difference frame-based event simulator.
class SparseInterpolatedEventSimulator:
    def __init__(self, 
                 optical_flow_calculator,
                 num_inter_frames: int, 
                 c_pos: int, # Positive threshold
                 c_neg: int  # Negative threshold
                 ):
        self.optical_flow_calculator = optical_flow_calculator
        self.num_inter_frames = num_inter_frames
        self.c_pos_ = c_pos
        self.c_neg_ = c_neg
        self.name_ = 'SparseInterpolatedEventSimulator'
        self.out_frames_ = []

    # Events
    def getEvents(self,
                  prev_frame:np.array, # Frame form the previous time step
                  frame:np.array,      # Frame form the current time step
                  prev_timestamp:int, # Time stamp of the previous time step
                  timestamp:int,      # Time stamp of the current time step
                  num_frames:int          # Number of frames
                  ):
        num_frames = 1
        c_pos = self.c_pos_
        mask = cv2.absdiff(prev_frame, frame)
        ret, mask = cv2.threshold(mask, c_pos, 255, cv2.THRESH_BINARY)
        
        prev_points = cv2.findNonZero(mask)
        next_points = self.optical_flow_calculator.calculateFlow(prev_frame, frame, prev_points)

        prev_inter_frame = prev_frame

        darker_frame = np.zeros_like(prev_frame, dtype=np.uint8)
        lighter_frame = np.zeros_like(prev_frame, dtype=np.uint8)
        
        x_res, y_res, channel = mask.shape
        
        for j in range(num_inter_frames+2):
            alpha = (float)(j / (num_inter_frames + 1))
            current_timestamp = prev_timestamp + alpha * (timestamp - prev_timestamp)
            inter_frame = prev_frame.copy()

            for i in range(len(next_points)):
                inter_point = prev_points[i] + (next_points[i] - prev_points[i]) * alpha
                if interpoint.x > (x_res - 1) or inter_point.y > (y_res - 1) or interpoint.x < 0 or inter_point.y < 0:
                    continue
                inter_frame[inter_point.x, inter_point.y] = prev_frame[prev_points[i].x, prev_points[i].y]
            
            darker = cv2.subtract(prev_inter_frame, inter_frame)
            lighter = cv2.subtract(inter_frame, prev_inter_frame)

            darker = cv2.threshold(darker, c_neg, 255, cv2.THRESH_BINARY)
            lighter = cv2.threshold(lighter, c_pos, 255, cv2.THRESH_BINARY)

            packEvents(lighter, darker, current_timestamp, events_)

        return events_

    # Accumulated Events
    def getEventFrame(self, 
                      prev_frame,
                      frame,
                    #   num_frames:int
                      ):
        # self.out_frames_ = [None] * num_frames
        num_frames = 1
        num_inter_frames = self.num_inter_frames

        mask = cv2.absdiff(prev_frame, frame)
        ret, mask = cv2.threshold(mask, self.c_pos_, 255, cv2.THRESH_BINARY)
        
        prev_points = cv2.findNonZero(mask)
        next_points = self.optical_flow_calculator.calculateFlow(prev_frame, frame, prev_points)
        # next_points : (N, 1, 2) where N = 5132

        prev_inter_frame = prev_frame

        darker_frame = np.zeros_like(prev_frame, dtype=np.uint8)
        lighter_frame = np.zeros_like(prev_frame, dtype=np.uint8)
        
        x_res, y_res = mask.shape
        # x_res, y_res, channels = mask.shape
        
        for j in range( num_inter_frames + 2 ):
            alpha = float(j) / float(num_inter_frames + 1)
            inter_frame = prev_frame.copy()
            for i in range(len(next_points)):
                inter_point = prev_points[i] + (next_points[i] - prev_points[i]) * alpha

                prev_points_x = prev_points[i][0][1]
                prev_points_y = prev_points[i][0][0]
  
                inter_point_x = inter_point[0][1]
                inter_point_y = inter_point[0][0]
                
                if inter_point_x > (x_res - 1) or inter_point_y > (y_res - 1) or inter_point_x < 0 or inter_point_y < 0:
                    continue
                inter_frame[int(inter_point_x)][int(inter_point_y)]= prev_frame[int(prev_points_x)][int(prev_points_y)]
            
            darker = cv2.subtract(prev_inter_frame, inter_frame) # (h,w)
            lighter = cv2.subtract(inter_frame, prev_inter_frame)
            
            cv2.add(darker_frame, darker, darker_frame)
            cv2.add(lighter, lighter_frame, lighter_frame)

            _, darker_frame = cv2.threshold(darker_frame, self.c_neg_, 255, cv2.THRESH_BINARY)
            _, lighter_frame = cv2.threshold(lighter_frame, self.c_pos_, 255, cv2.THRESH_BINARY)

            prev_inter_frame = inter_frame
        zeros = np.zeros_like(prev_frame, dtype=np.uint8)
        channels = [lighter_frame, zeros, darker_frame]
        self.out_frames_.append(cv2.merge(channels))
        
        return self.out_frames_

   
    # Name of th Event Simulator
    def getName(self):
        return str(self.name_) + '_' + str(self.c_pos_) + '_' + str(self.c_neg_)
    

    # def SparseSim(self):
    #     events = self.getEvents(prev_frame, frame, prev_timestamp, timestamp)
    #     event_frames = self.out_frames_(prev_frame, frame)
    #     print(T)


def main():
    pass
if __name__ == '__main__':
    main()

