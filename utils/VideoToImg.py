"""
    Project Name: VideoToImg.py
    Description : Capture Frame of the Video and Save it
    Author: Youngjin Moon
    Date: 
        First Release: 23.08.14
"""

import cv2
import os
import pdb

def main():
    filepath = 'dataset\sample.mp4'
    out_path = 'dataset'
    video = cv2.VideoCapture(filepath) 

    if not video.isOpened():
        print("Could not Open :", filepath)
        exit(0)

    length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv2.CAP_PROP_FPS)
    check_num = 1
    print("length :", length)
    print("width :", width)
    print("height :", height)
    print("fps :", fps)

    count = 0
    while(video.isOpened()):
        ret, image = video.read()
        print(count)
        if(int(video.get(1)) % int(fps) == 0): # Save by 1 sec
            if count == check_num:
                cv2.imwrite(out_path + "/prev_frame.png", image)
                print('Saved frame number :', str(int(video.get(1))))
            elif count == check_num+1:
                cv2.imwrite(out_path + "/frame.png", image)
                print('Saved frame number :', str(int(video.get(1))))
            count += 1
        
    video.release()

if __name__ == '__main__':
    main()
  
  



