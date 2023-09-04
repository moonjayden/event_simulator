"""
    Project Name: DenseInterpolatedEventSimulator.py
    Description : 
    Author: Youngjin Moon
    Date: 
        First Release: 23.08.16
"""
import cv2
from typing import List, Tuple
import numpy as np
import pdb 

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


# Implementing dense interpolated-based event simulator.
class DenseInterpolatedEventSimulator:
    def __init__(self, 
                 optical_flow_calculator, 
                 num_inter_frames, 
                 c_pos, # Positive threshold
                 c_neg  # Negative threshold
                 ):
        self.num_inter_frames = num_inter_frames
        self.optical_flow_calculator = optical_flow_calculator
        self.name = "DenseInterpolatedEventSimulator"
        self.c_pos_ = c_pos
        self.c_neg_ = c_neg
        self.events_ = []
        self.out_frames_ = []

    # Events
    def getEvents(self,
                  prev_frame:np.array, # Frame form the previous time step
                  frame:np.array,      # Frame form the current time step
                  prev_timestamp:int, # Time stamp of the previous time step
                  timestamp:int      # Time stamp of the current time step
                  ):
        flow = self.optical_flow_calculator.calculateFlow(prev_frame, frame)
        prev_inter_frame = prev_frame
        self.events_.clear()
        darker_frames = []
        lighter_frames = []
        
        for j in range(self.num_inter_frames + 2):
            alpha = float(j) / float(self.num_inter_frames + 1)
            current_timestamp = prev_timestamp + alpha * (timestamp - prev_timestamp)
            interflow = alpha * flow
            map = np.zeros_like(flow, dtype=np.complex64)
            
            for y in range(map.shape[0]):
                for x in range(map.shape[1]):
                    f = interflow[y, x]
                    # map[y, x] = [x + f[1], y + f[0]]
                    map[y, x] = x + f[0] + (y + f[1])*1j

            flow_parts = cv2.split(map)
            inter_frame = cv2.remap(prev_frame, flow_parts[0], flow_parts[1], cv2.INTER_LINEAR)

            darker = cv2.subtract(prev_inter_frame, inter_frame)
            lighter = cv2.subtract(inter_frame, prev_inter_frame)
            _, lighter = cv2.threshold(lighter, self.c_pos_, 255, cv2.THRESH_BINARY)
            _, darker = cv2.threshold(darker, self.c_neg_, 255, cv2.THRESH_BINARY)
            
            packEvents(lighter, darker, current_timestamp, self.events_)
            prev_inter_frame = inter_frame

        return self.events_


    # Accumulated Events
    def getEventFrame(self, 
                      prev_frame,
                      frame,
                      ):
        flow = self.optical_flow_calculator.calculateFlow(prev_frame, frame)
        prev_inter_frame = prev_frame
        darker_frame = np.zeros_like(prev_frame, dtype=np.uint8)
        lighter_frame = np.zeros_like(prev_frame, dtype=np.uint8)
        # darker_frames = []
        # lighter_frames = []
        import time
        for j in range(self.num_inter_frames + 2):
            f_s = time.time()
            alpha = float(j) / float(self.num_inter_frames + 1)
            interflow = alpha * flow
            map = np.zeros_like(flow)
            # map = np.zeros_like(flow, dtype=np.complex64)
            x_res, y_res, channel = map.shape
            s_s = time.time()
            for y in range(map.shape[0]):
                for x in range(map.shape[1]):
                    f = interflow[y, x]
                    map[y, x] = [x + f[0], y + f[1]]
                    # map[y, x] = x + f[0] + (y + f[1])*1j
            s_e = time.time()
            # pdb.set_trace()
            # h, w = map.shape[:2]
            # map2, map1 = np.indices((h, w), dtype=np.float32)
            # inter_frame = cv2.remap(prev_frame, map1, map2, cv2.INTER_LINEAR)
            inter_frame = cv2.remap(prev_frame, map, None, cv2.INTER_LINEAR)

            darker = cv2.subtract(prev_inter_frame, inter_frame)
            lighter = cv2.subtract(inter_frame, prev_inter_frame)
            _, lighter = cv2.threshold(lighter, self.c_pos_, 255, cv2.THRESH_BINARY)
            _, darker = cv2.threshold(darker, self.c_neg_, 255, cv2.THRESH_BINARY)

            darker_frame = cv2.add(darker, darker_frame)
            lighter_frame = cv2.add(lighter, lighter_frame)
            prev_inter_frame = inter_frame
            f_e = time.time()

            print(f'first loop = {1000*(f_e-f_s)}ms    second loop = {1000*(s_e-s_s)}ms')

        self.out_frames_.append(cv2.merge([darker_frame, np.zeros_like(prev_frame, dtype=np.uint8), lighter_frame]))

        return self.out_frames_
   
    # Name of th Event Simulator
    def getName(self):
        return str(self.name_) + '_' + str(self.c_pos_) + '_' + str(self.c_neg_)
    


def main():
    pass

if __name__ == '__main__':
    main()


