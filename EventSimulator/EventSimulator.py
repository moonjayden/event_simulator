"""
    Project Name: EventSimulator.py
    Description : Basic Code Frame of EventSimulator (Not Related with main)
    Author: Youngjin Moon
    Date: 
        First Release: 23.08.14
"""
"""
    Checked List
    -def packEvents

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
def packEvents(lighter_events:cv2.Mat, # cv::Mat containing pixel which got brighter
                darker_events:cv2.Mat, # cv::Mat containing pixel which got darker
                timestamp:int          # Time stamp of the events
                ) -> List[Event]:
    pos_polarity_events = np.transpose(np.nonzero(lighter_events))
    neg_polarity_events = np.transpose(np.nonzero(darker_events))
    # pdb.set_trace()

    output = []

    for point in neg_polarity_events:
        pdb.set_trace()
        output.append(Event(point[1], point[0], timestamp, False))
        
    for point in pos_polarity_events:
        output.append(Event(point[1], point[0], timestamp, True))

    return output


# Base class for event simulators. 
class EventSimulator:
    """
    def __init__(self, prev_frame:cv2.Mat, # Frame form the previous time step
                  frame:cv2.Mat,      # Frame form the current time step
                  prev_timestamp:int, # Time stamp of the previous time step
                  timestamp:int,      # Time stamp of the current time step
                  num_frames          # Number of frames
                  ):
        self.prev_frame = prev_frame
        self.frame = frame
        self.prev_timestamp = prev_timestamp
        self.timestamp = timestamp
        self.num_frames = num_frames
    """
    # Returns the simulated events from the simulator
    def getEvents(self, 
                  prev_frame:cv2.Mat, # Frame form the previous time step
                  frame:cv2.Mat,      # Frame form the current time step
                  prev_timestamp:int, # Time stamp of the previous time step
                  timestamp:int,      # Time stamp of the current time step
                  num_frames          # Number of frames
                  )-> List[Event]:
        lighter_events = cv2.absdiff(frame, prev_frame)
        darker_events = cv2.absdiff(prev_frame, frame)
        events = packEvents(lighter_events, darker_events,num_frames)
        return events

    # Returns an accumulated event frame given the events from the simulator
    def getEventFrame(self, 
                      prev_frame:cv2.Mat, 
                      frame:cv2.Mat, 
                      num_frames) -> List[Event]:
        pass
        # event_frames = 
        # return event_frames

    # Set the frame size
    def setup(self, frame_size):
        self.frame_size_ = frame_size

    # Returns the name of the event simulator
    def getName(self):
        pass


"""
def main():
    # test
    lighter_events = np.array([[0, 1, 0],
                            [1, 0, 1],
                            [0, 1, 0]], dtype=np.uint8)
    darker_events = np.array([[0, 1, 0],
                            [1, 0, 1],
                            [0, 1, 0]], dtype=np.uint8)
    timestamp = 100

    packEvents(lighter_events, darker_events, timestamp)


if __name__ == '__main__':
    main()
"""
