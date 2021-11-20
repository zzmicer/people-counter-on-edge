import cv2
import numpy as np


def preprocess(input_image, height, width):
    '''
    Given an input image, height and width:
    - Resize to height and width
    - Transpose the final "channel" dimension to be first
    - Reshape the image to add a "batch" of 1 at the start 
    '''
    image = cv2.resize(input_image, (width, height))
    image = image.transpose((2,0,1))
    image = image.reshape(1, 3, height, width)
    return image


def draw_box(frame, info):
    '''
    Draw boxes around a person in the frame.
    '''
    H,W,_ = frame.shape
    _, _, _, x1, y1, x2, y2 = info

    x1 = int(W * x1)
    x2 = int(W * x2)
    y1 = int(H * y1)
    y2 = int(H * y2)

    
    frame = cv2.rectangle(frame, (x1,y1), (x2,y2), (255,0,0), 2)

    return frame
