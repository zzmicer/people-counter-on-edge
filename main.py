"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


import os
import sys
import time
import socket
import json
import cv2
import numpy as np

import logging as log
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network

from src.utils import preprocess,draw_box

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60

# Check for a person detection every 10 frames
PERSON_FRAMERATE = 10


def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to video file")
    
    # Note - CPU extensions are moved to plugin since OpenVINO release 2020.1. 
    # The extensions are loaded automatically while     
    # loading the CPU plugin, hence 'add_extension' need not be used.
    #     parser.add_argument("-l", "--cpu_extension", required=False, type=str,
    #                         default=None,
    #                         help="MKLDNN (CPU)-targeted custom layers."
    #                              "Absolute path to a shared library with the"
    #                              "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    return parser


def connect_mqtt():
    ### TODO: Connect to the MQTT client ###
    client = None
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)
    return client


def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    # Initialise the class
    infer_network = Network()
    # Set Probability threshold for detections
    prob_threshold = args.prob_threshold

    ### Load the model through `infer_network` ###
    infer_network.load_model(args.model, args.device)
    net_input_shape = infer_network.get_input_shape()
    n, c, h, w = net_input_shape

    if(args.input == '0'):
        args.input = 0
    cap = cv2.VideoCapture(args.input)
    cap.open(args.input)

    #Init info
    count_history = [0]*PERSON_FRAMERATE
    current_count = 0
    total_people = 0
    counting = False
    delta = 0

    while cap.isOpened():
        ### Read from the video capture ###
        flag, frame = cap.read()
        if not flag:
            break
        key_pressed = cv2.waitKey(60)
        cv2.imshow("Input", frame)
        
        resized_img = preprocess(frame,h,w)
        
        infer_network.exec_net(resized_img)

        if infer_network.wait() == 0:
            result = infer_network.get_output()
            
            result = result.squeeze()
            # Handle only person info
            result = result[result[:,1]==1]
            result = result[result[:,2]>=args.prob_threshold]
            
            # Number of persons in current frame
            person_in_frame = result.shape[0]

            # Update history
            count_history.pop(0)
            count_history.append(person_in_frame)

            previous_count = current_count
            current_count = max(set(count_history), key = count_history.count)
            delta = current_count - previous_count
            if delta>0:
                # update number of people seen
                total_people+=delta

            print(total_people)

            # Draw bbox around detected person
            for person in result:
                frame = draw_box(frame, person)
            cv2.imshow("Output", frame)

            
        ### Publish statistics
        if client is not None:
            client.publish("person", json.dumps({"count": current_count, 'total': tot_people}))

        ### Send the frame to the FFMPEG server ###
        try:
            sys.stdout.buffer.write(frame)  
            sys.stdout.flush()
        except BrokenPipeError:
            print ('BrokenPipeError caught', file = sys.stderr)


        if key_pressed == 27:
            break
        # Release the capture and destroy any OpenCV windows
    
    sys.stderr.close()
    client.disconnect()

    cap.release()
    cv2.destroyAllWindows()

def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args, client)


if __name__ == '__main__':
    main()
