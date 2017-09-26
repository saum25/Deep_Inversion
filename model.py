#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Network architecture definition for Singing Voice Detection experiment.

Author: Jan Schlüter
"""

import lasagne
from lasagne.layers import (InputLayer, Conv2DLayer, MaxPool2DLayer,
                            DenseLayer, dropout)

def architecture(input_var, input_shape):
    network_trained = {}
    network_trained['input'] = InputLayer(input_shape, input_var)
    kwargs = dict(nonlinearity=lasagne.nonlinearities.leaky_rectify,
                  W=lasagne.init.Orthogonal())
    network_trained['conv1'] = Conv2DLayer(network_trained['input'], 64, 3, **kwargs)
    network_trained['conv2'] = Conv2DLayer(network_trained['conv1'], 32, 3, **kwargs)
    network_trained['mp3'] = MaxPool2DLayer(network_trained['conv2'], 3)
    network_trained['conv4'] = Conv2DLayer(network_trained['mp3'], 128, 3, **kwargs)
    network_trained['conv5'] = Conv2DLayer(network_trained['conv4'], 64, 3, **kwargs)
    network_trained['mp6'] = MaxPool2DLayer(network_trained['conv5'], 3)
    network_trained['fc7'] = DenseLayer(dropout(network_trained['mp6'], 0.5), 256, **kwargs)
    network_trained['fc8'] = DenseLayer(dropout(network_trained['fc7'], 0.5), 64, **kwargs)
    network_trained['fc9'] = DenseLayer(dropout(network_trained['fc8'], 0.5), 1,
                       nonlinearity=lasagne.nonlinearities.sigmoid,
                       W=lasagne.init.Orthogonal())    
    
    return network_trained
