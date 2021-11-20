#!/usr/bin/env python3
"""
 Copyright (c) 2018 Intel Corporation.

 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit persons to whom the Software is furnished to do so, subject to
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
import logging as log
from openvino.inference_engine import IENetwork, IECore
import ngraph as ng


class Network:
    """
    Load and configure inference plugins for the specified target devices 
    and performs synchronous and asynchronous modes for the specified infer requests.
    """

    def __init__(self):
        self.plugin = None
        self.network = None
        self.input_blob = None
        self.output_blob = None
        self.exec_network = None
        self.infer_request = None

    def load_model(self, model, device="CPU", cpu_extension=None):
        """
        Load the model given IR files.
        Defaults to CPU as device for use in the workspace.
        Synchronous requests made within.
        """

        model_xml = model
        model_bin = os.path.splitext(model_xml)[0] + ".bin"

        self.plugin = IECore()
        self.network = self.plugin.read_network(model=model_xml, weights=model_bin)

        if cpu_extension and "CPU" in device:
            self.plugin.add_extension(cpu_extension, device)

        # Check for supported layers
        supported_layers = self.plugin.query_network(
            network=self.network, device_name=device)

        # Check for any unsupported layers, and let the user
        # know if anything is missing. Exit the program, if so.
        net_layers = []
        ngraph_func = ng.function_from_cnn(self.network)
        for op in ngraph_func.get_ordered_ops():
            net_layers.append(op.get_friendly_name())

        for layer in net_layers:
            if layer in supported_layers.keys():
                continue
            else:
                print("Layer {} is unsupported".format(str(layer)))
                print("Check whether extensions are available to add to IECore.")
                exit(1)

        self.exec_network = self.plugin.load_network(self.network, device)
        self.input_blob = next(iter(self.network.inputs))
        self.output_blob = next(iter(self.network.outputs))

        return 

    def get_input_shape(self):
        """
        Gets the input shape of the network
        """
        return self.network.inputs[self.input_blob].shape
    
    def get_output_shape(self):
        """
        Gets the output shape of the network
        """
        return self.network.inputs[self.output_blob].shape


    def exec_net(self, image):
        """
        Makes an asynchronous inference request, given an input image.
        """
        self.infer_request = self.exec_network.start_async(
            request_id=0, inputs={self.input_blob: image})
        return

    def wait(self):
        """
        Checks the status of the inference request.
        """
        status = self.infer_request.wait()
        return status

    def get_output(self):
        """
        Returns a list of the results for the output layer of the network
        """
        return self.infer_request.outputs[self.output_blob]
