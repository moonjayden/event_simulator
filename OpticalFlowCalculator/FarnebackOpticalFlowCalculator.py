"""
    Project Name: FarnebackOpticalFlowCalcultor.py
    Description : Optical Flow Calculator (Farneback Method)
    Author: Youngjin Moon
    Date: 
        First Release: 23.08.17
"""
import cv2
from typing import List, Tuple
import numpy as np

class FarnebackOpticalFlowCalculator:
    def __init__(self):
        self.name_ = "FarnebackOpticalFlowCalculator"
    
    def calculateFlow(self, prev_frame, frame, prev_points = None):
        flow = None
        flow = cv2.calcOpticalFlowFarneback(prev_frame, frame, flow, 0.5 ,3, 15, 3, 5, 1.2, 0)
        return flow
    
    def getName(self):
        return self.name_