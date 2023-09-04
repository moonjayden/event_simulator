"""
    Project Name: main.py
                                                   DIS / Farneback               LK                    LK
    Description : Main File of Event Simulator(DenseFrameInterpolated / SparseFrameInterpolated / Difference Frame)
    Author: Youngjin Moon
    Date: 
        First Release: 23.08.11
"""
import cv2
import numpy as np
from EventSimulator.BasicDifferenceEventSimulator import *
from EventSimulator.DenseInterpolatedEventSimulator import *
from EventSimulator.SparseInterpolatedEventSimulator import *
from EventSimulator.DifferenceInterpolatedEventSimulator import *

# from OpticalFlowCalculator.DISOpticalFlowCalculator import *
from OpticalFlowCalculator.LKOpticalFlowCalculator import *
from OpticalFlowCalculator.FarnebackOpticalFlowCalculator import *
from Player import *

def main():
    
    # Resolution of the video frame
    x_res, y_res = 640, 480

    # Time in milliseconds to wait between displaying frames
    walt_time_ms = 0

    # Positive and negative contrast values, number of intermediate frames, and division factor.
    C_pos, C_neg = 20, 20
    num_inter_frames = 10
    div_factor = 10

    # Offset value for contrast adjustment.
    C_offset = 10

    # Dataset
    video_path = "./dataset/sample_high.mp4"
    # video_path = "./dataset/sample_low.mp4"

    """
    ------------------------
    Optical Flow Calculator
    ------------------------
    """
    # Creating an instance of DISOpticalFlowCalculator
    ofc_dis_low = DISOpticalFlowCalculator(DISOpticalFlowQuality.LOW)
    # ofc_dis_med = DISOpticalFlowCalculator(DISOpticalFlowQuality.MEDIUM)
    # ofc_dis_high = DISOpticalFlowCalculator(DISOpticalFlowQuality.HIGH)
    # ofc_dis_ext = DISOpticalFlowCalculator(DISOpticalFlowQuality.EXTREME)

    # Creating an instance of DISOpticalFlowCalculator 
    ofc_farne = FarnebackOpticalFlowCalculator()
    
    # Creating an instance of LKOpticalFlowCalculator
    ofc_lk = LKOpticalFlowCalculator()

    """
    ------------------------
    Event Simulator
    ------------------------
    """

# BasicDifferenceEventSimulator
    bdes = BasicDifferenceEventSimulator(C_pos, C_neg)
    
# DenseInterpolatedEventSimulator w. DIS optical flow, number of intermediate frames, adjusted contrast values.
    dies = DenseInterpolatedEventSimulator(ofc_dis_low, num_inter_frames, C_pos // div_factor, C_neg // div_factor)
    # dies = DenseInterpolatedEventSimulator(ofc_dis_med, num_inter_frames, C_pos // div_factor, C_neg // div_factor)
    # dies = DenseInterpolatedEventSimulator(ofc_dis_high, num_inter_frames, C_pos // div_factor, C_neg // div_factor)
    # dies = DenseInterpolatedEventSimulator(ofc_dis_ext, num_inter_frames, C_pos // div_factor, C_neg // div_factor)
    
# DenseInterpolatedEventSimulator w. Farneback optical flow, number of intermediate frames, adjusted contrast values.
    # dies = DenseInterpolatedEventSimulator(ofc_farne, num_inter_frames, C_pos // div_factor, C_neg // div_factor)

# SparseInterpolatedEventSimulator w. LK optical flow, number of intermediate frames, original and adjusted contrast values, and an offset.
    spies = SparseInterpolatedEventSimulator(ofc_lk, num_inter_frames, C_pos, C_neg)
    # spies = SparseInterpolatedEventSimulator(ofc_lk, num_inter_frames, C_pos + C_offset, C_neg + C_offset)

# DifferenceInterpolatedEventSimulator w. LK optical flow, number of intermediate frames, original and adjusted contrast values, and an offset.
    dfies = DifferenceInterpolatedEventSimulator(ofc_lk, num_inter_frames, C_pos, C_neg, num_inter_frames, num_inter_frames)
    """
    ------------------------
    Video Player
    ------------------------
    """
# Create VideoStreamer instance w. BasicDifferenceEventSimulator.
    # s= time.time()
    # streamer = VideoStreamer(bdes)
    # streamer.simulate_from_stream(video_path)
    # e= time.time()
    # print(f'bdes = {1000*(e-s)}ms')

# Create VideoStreamer instance w. DenseInterpolatedEventSimulator.
    # s= time.time()
    streamer = VideoStreamer(dies)
    streamer.simulate_from_stream(video_path)
    # e= time.time()
    # print(f'dies = {1000*(e-s)}ms')

# Create VideoStreamer instance w. SparseInterpolatedEventSimulator.
    # s= time.time()
    # streamer = VideoStreamer(spies)
    # streamer.simulate_from_stream(video_path)
    # e= time.time()
    # print(f'spies = {1000*(e-s)}ms')

# Create VideoStreamer instance w. DifferenceInterpolatedEventSimulator.
    # s= time.time()
    # streamer = VideoStreamer(dfies,Diff=True)
    # streamer.simulate_from_stream(video_path)    
    # e= time.time()
    # print(f'dfies = {1000*(e-s)}ms')

# Initiate the streaming process, starting from frame 0.
    # streamer.stream(0)

    """     Dummy
    ------------------------
    # SparseInterpolatedFrameEventSimulator w. LK optical flow, number of intermediate frames, adjusted contrast values.
    ssmes = SparseInterpolatedFrameEventSimulator(ofc_lk, num_inter_frames, C_pos // 2, C_neg // 2)
    
    # provide the path to a video file to use the OpenCVPlayer.
    video_path = "../res/videos/car.mp4"
    cv_player = OpenCVPlayer(dsmes, wait_time_ms)
    cv_player.play(video_path)
    ------------------------
    """


if __name__ == '__main__':
    main()

