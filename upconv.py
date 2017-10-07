'''
Created on 26 Aug 2017

@author: Saumitra
'''

import lasagne
from lasagne.layers import (InputLayer, DenseLayer, ReshapeLayer, TransposedConv2DLayer, batch_norm)

def architecture_upconv(input_var, input_shape):
    
    net = {}
    
    net['data'] = InputLayer(input_shape, input_var)
    #net['fc1'] = DenseLayer(net['data'], num_units=64, W=lasagne.init.Orthogonal(), nonlinearity=lasagne.nonlinearities.very_leaky_rectify)
    net['fc2'] = batch_norm(DenseLayer(net['data'], num_units=256, W=lasagne.init.HeNormal(), nonlinearity=lasagne.nonlinearities.elu))
    net['rs1'] = ReshapeLayer(net['fc2'], (32, 16, 4, 4)) # assuming that the shape is batch x depth x row x columns
    kwargs = dict(filter_size= 4,
                  stride = 2,
                  crop = 1,
                  nonlinearity=lasagne.nonlinearities.elu,
                  W=lasagne.init.HeNormal())
    net['uc1'] = batch_norm(TransposedConv2DLayer(net['rs1'], num_filters= 16, **kwargs))
    #print(net['uc1'].output_shape)    
    net['uc2'] = batch_norm(TransposedConv2DLayer(net['uc1'], num_filters= 8, **kwargs))
    #print(net['uc2'].output_shape)    
    net['uc3'] = batch_norm(TransposedConv2DLayer(net['uc2'], num_filters= 4, **kwargs))
    net['uc4'] = batch_norm(TransposedConv2DLayer(net['uc3'], num_filters= 2, **kwargs))
    # In Jan's GAN code no batch_norm in the last layer: Need to find why?
    net['uc5'] = TransposedConv2DLayer(net['uc4'], num_filters= 1, **kwargs)

    # slicing the output to 115 x 80 size
    #print(net['uc5'].output_shape)    
    net['s1'] = lasagne.layers.SliceLayer(net['uc5'], slice(0, 115), axis=-2)
    net['out'] = lasagne.layers.SliceLayer(net['s1'], slice(0, 80), axis=-1)
    #print(net['out'] .output_shape)
    
    return net['out']
