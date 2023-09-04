"""
    Project Name: LKOpticalFlowCalcultor.py
    Description : Optical Flow Calculator (LK Method)
    Author: Youngjin Moon
    Date: 
        First Release: 23.08.17
"""
import cv2
import numpy as np
import math

class LKOpticalFlowCalculator:
    def __init__(self, max_level=3, window_size=15):
        self.max_level_ = max_level
        self.window_size_ = window_size
        self.name_ = "LKOpticalFlowCalculator"
    
    def calculateFlow(self, prev_frame, frame, points):
        # if len(points)==0:
        #     return cv2.Point2f()
        
        points_dtype = points.astype(np.float32)
        new_points = None
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 10, 0.03)

        new_points, status, err = cv2.calcOpticalFlowPyrLK(prev_frame, frame, points_dtype, new_points, 
                                winSize=(self.window_size_, self.window_size_),
                                maxLevel=self.max_level_, criteria=criteria)
        return new_points
    
    def getName(self):
        return self.name_