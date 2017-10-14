'''
Created on 26 Aug 2017

@author: Saumitra
'''

import lasagne
from lasagne.layers import (InputLayer, DenseLayer, ReshapeLayer, TransposedConv2DLayer, batch_norm, Conv2DLayer)

def architecture_upconv_fc8(input_var, input_shape):
    
    net = {}
    
    net['data'] = InputLayer(input_shape, input_var)
    print(net['data'].output_shape)
    print("\r printing layer's output shape")
    net['fc1'] = batch_norm(DenseLayer(net['data'], num_units=64, W=lasagne.init.HeNormal(), nonlinearity=lasagne.nonlinearities.elu))
    print(net['fc1'].output_shape)
    net['fc2'] = batch_norm(DenseLayer(net['fc1'], num_units=256, W=lasagne.init.HeNormal(), nonlinearity=lasagne.nonlinearities.elu))
    print(net['fc2'].output_shape)
    net['rs1'] = ReshapeLayer(net['fc2'], (32, 16, 4, 4)) # assuming that the shape is batch x depth x row x columns
    
    kwargs = dict(nonlinearity=lasagne.nonlinearities.elu,
                  W=lasagne.init.HeNormal())
    
    net['uc1'] = batch_norm(TransposedConv2DLayer(net['rs1'], num_filters= 16, filter_size= 4, stride = 2, crop=1, **kwargs))
    print(net['uc1'].output_shape)
    net['c1'] = batch_norm(Conv2DLayer(net['uc1'], num_filters= 16, filter_size= 3, stride = 1, pad=1, **kwargs))
    print(net['c1'].output_shape)    
    
    net['uc2'] = batch_norm(TransposedConv2DLayer(net['c1'], num_filters= 8, filter_size= 4, stride = 2, crop=1, **kwargs))
    print(net['uc2'].output_shape)
    net['c2'] = batch_norm(Conv2DLayer(net['uc2'], num_filters= 8, filter_size= 3, stride = 1, pad=1, **kwargs))
    print(net['c2'].output_shape)    
    
    net['uc3'] = batch_norm(TransposedConv2DLayer(net['c2'], num_filters= 4, filter_size= 4, stride = 2, crop=1, **kwargs))
    print(net['uc3'].output_shape)
    net['c3'] = batch_norm(Conv2DLayer(net['uc3'], num_filters= 4, filter_size= 3, stride = 1, pad=1, **kwargs))
    print(net['c3'].output_shape)

    net['uc4'] = batch_norm(TransposedConv2DLayer(net['c3'], num_filters= 2, filter_size= 4, stride = 2, crop=1, **kwargs))
    print(net['uc4'].output_shape)
    net['c4'] = batch_norm(Conv2DLayer(net['uc4'], num_filters= 2, filter_size= 3, stride = 1, pad=1, **kwargs))
    print(net['c4'].output_shape)
    
    net['uc5'] = TransposedConv2DLayer(net['c4'], num_filters= 1, filter_size= 4, stride = 2, crop=1, **kwargs)
    print(net['uc5'].output_shape)

    # slicing the output to 115 x 80 size
    net['s1'] = lasagne.layers.SliceLayer(net['uc5'], slice(0, 115), axis=-2)
    net['out'] = lasagne.layers.SliceLayer(net['s1'], slice(0, 80), axis=-1)
    print(net['out'] .output_shape)
    
    print("\r Number of parameter to be learned: %d" %(lasagne.layers.count_params(net['out'])))
    
    return net['out']

def architecture_upconv_fc7(input_var, input_shape):
    
    net = {}
    
    net['data'] = InputLayer(input_shape, input_var)
    print(net['data'].output_shape)
    net['fc2'] = batch_norm(DenseLayer(net['data'], num_units=256, W=lasagne.init.HeNormal(), nonlinearity=lasagne.nonlinearities.elu))
    print(net['fc2'].output_shape)
    
    net['rs1'] = ReshapeLayer(net['fc2'], (32, 16, 4, 4)) # assuming that the shape is batch x depth x row x columns
    
    kwargs = dict(nonlinearity=lasagne.nonlinearities.elu,
                  W=lasagne.init.HeNormal())
    
    net['uc1'] = batch_norm(TransposedConv2DLayer(net['rs1'], num_filters= 16, filter_size= 4, stride = 2, crop=1, **kwargs))
    print(net['uc1'].output_shape)
    net['c1'] = batch_norm(Conv2DLayer(net['uc1'], num_filters= 16, filter_size= 3, stride = 1, pad=1, **kwargs))
    print(net['c1'].output_shape)    
    
    net['uc2'] = batch_norm(TransposedConv2DLayer(net['c1'], num_filters= 8, filter_size= 4, stride = 2, crop=1, **kwargs))
    print(net['uc2'].output_shape)
    net['c2'] = batch_norm(Conv2DLayer(net['uc2'], num_filters= 8, filter_size= 3, stride = 1, pad=1, **kwargs))
    print(net['c2'].output_shape)    
    
    net['uc3'] = batch_norm(TransposedConv2DLayer(net['c2'], num_filters= 4, filter_size= 4, stride = 2, crop=1, **kwargs))
    print(net['uc3'].output_shape)
    net['c3'] = batch_norm(Conv2DLayer(net['uc3'], num_filters= 4, filter_size= 3, stride = 1, pad=1, **kwargs))
    print(net['c3'].output_shape)

    net['uc4'] = batch_norm(TransposedConv2DLayer(net['c3'], num_filters= 2, filter_size= 4, stride = 2, crop=1, **kwargs))
    print(net['uc4'].output_shape)
    net['c4'] = batch_norm(Conv2DLayer(net['uc4'], num_filters= 2, filter_size= 3, stride = 1, pad=1, **kwargs))
    print(net['c4'].output_shape)
    
    net['uc5'] = TransposedConv2DLayer(net['c4'], num_filters= 1, filter_size= 4, stride = 2, crop=1, **kwargs)
    print(net['uc5'].output_shape)

    # slicing the output to 115 x 80 size
    net['s1'] = lasagne.layers.SliceLayer(net['uc5'], slice(0, 115), axis=-2)
    net['out'] = lasagne.layers.SliceLayer(net['s1'], slice(0, 80), axis=-1)
    print(net['out'] .output_shape)
    
    print("\r Number of parameter to be learned: %d" %(lasagne.layers.count_params(net['out'])))
    
    return net['out']

def architecture_upconv_c5_mp6(input_var, input_shape):
    
    net = {}
    
    kwargs = dict(nonlinearity=lasagne.nonlinearities.elu,
                  W=lasagne.init.HeNormal())
    
    net['data'] = InputLayer(input_shape, input_var)
    print(net['data'].output_shape)
    print("\r Layer-by-layer output shapes of the upconvolutional network")
    
    # Bunch of 3 x 3 convolution layers: experimentally we found that, adding 3 conv layers in start than in middle is better: but why?
    '''net['c1'] = batch_norm(Conv2DLayer(net['data'], num_filters= 64, filter_size= 3, stride = 1, pad=1, **kwargs))
    print(net['c1'].output_shape)   
    net['c2'] = batch_norm(Conv2DLayer(net['c1'], num_filters= 64, filter_size= 3, stride = 1, pad=1, **kwargs))
    print(net['c2'].output_shape)
    net['c3'] = batch_norm(Conv2DLayer(net['c2'], num_filters= 64, filter_size= 3, stride = 1, pad=1, **kwargs))
    print(net['c3'].output_shape)
    net['c4'] = batch_norm(Conv2DLayer(net['c3'], num_filters= 64, filter_size= 3, stride = 1, pad=1, **kwargs))
    print(net['c3'].output_shape)'''
    
    # Bunch of transposed convolution layers  
    net['uc1'] = batch_norm(TransposedConv2DLayer(net['data'], num_filters= 64, filter_size= 4, stride = 2, crop=1, **kwargs))
    print(net['uc1'].output_shape)
    net['c1'] = batch_norm(Conv2DLayer(net['uc1'], num_filters= 64, filter_size= 3, stride = 1, pad=1, **kwargs))
    print(net['c1'].output_shape)
    
    net['uc2'] = batch_norm(TransposedConv2DLayer(net['c1'], num_filters= 32, filter_size= 4, stride = 2, crop=1, **kwargs))
    print(net['uc2'].output_shape)
    net['c2'] = batch_norm(Conv2DLayer(net['uc2'], num_filters= 32, filter_size= 3, stride = 1, pad=1, **kwargs))
    print(net['c2'].output_shape) 
    
    net['uc3'] = batch_norm(TransposedConv2DLayer(net['c2'], num_filters= 16, filter_size= 4, stride = 2, crop=1, **kwargs))
    print(net['uc3'].output_shape)
    net['c3'] = batch_norm(Conv2DLayer(net['uc3'], num_filters= 16, filter_size= 3, stride = 1, pad=1, **kwargs))
    print(net['c3'].output_shape)

    net['uc4'] = batch_norm(TransposedConv2DLayer(net['c3'], num_filters= 1, filter_size= 4, stride = 2, crop=1, **kwargs))
    print(net['uc4'].output_shape)
    '''net['c4'] = batch_norm(Conv2DLayer(net['uc4'], num_filters= 2, filter_size= 3, stride = 1, pad=1, **kwargs))
    print(net['c4'].output_shape)'''
    
    '''net['uc5'] = TransposedConv2DLayer(net['c4'], num_filters= 1, filter_size= 4, stride = 2, crop=1, **kwargs)
    print(net['uc5'].output_shape)'''

    # slicing the output to 115 x 80 size  
    net['s1'] = lasagne.layers.SliceLayer(net['uc4'], slice(0, 115), axis=-2)
    print(net['s1'].output_shape)  
    net['out'] = lasagne.layers.SliceLayer(net['s1'], slice(0, 80), axis=-1)
    print(net['out'] .output_shape)
    print("Number of parameter to be learned: %d" %(lasagne.layers.count_params(net['out'])))
    
    return net['out']

def architecture_upconv_conv4(input_var, input_shape):
    
    net = {}
    
    kwargs = dict(nonlinearity=lasagne.nonlinearities.elu,
                  W=lasagne.init.HeNormal())
    
    net['data'] = InputLayer(input_shape, input_var)
    print("\rLayer output shapes")
    print(net['data'].output_shape)
    
    # Bunch of 3 x 3 convolution layers: experimentally we found that, adding 3 conv layers in start than in middle is better: but why?
    net['c1'] = batch_norm(Conv2DLayer(net['data'], num_filters= 128, filter_size= 3, stride = 1, pad=1, **kwargs))
    print(net['c1'].output_shape)   
    '''net['c2'] = batch_norm(Conv2DLayer(net['c1'], num_filters= 128, filter_size= 3, stride = 1, pad=1, **kwargs))
    print(net['c2'].output_shape)
    net['c3'] = batch_norm(Conv2DLayer(net['c2'], num_filters= 128, filter_size= 3, stride = 1, pad=1, **kwargs))
    print(net['c3'].output_shape)'''
    
    # Bunch of transposed convolution layers  
    net['uc1'] = batch_norm(TransposedConv2DLayer(net['c1'], num_filters= 64, filter_size= 4, stride = 2, crop=1, **kwargs))
    print(net['uc1'].output_shape)
    
    net['uc2'] = batch_norm(TransposedConv2DLayer(net['uc1'], num_filters= 1, filter_size= 4, stride = 2, crop=1, **kwargs))
    print(net['uc2'].output_shape)

    # slicing the output to 115 x 80 size  
    net['s1'] = lasagne.layers.SliceLayer(net['uc2'], slice(0, 115), axis=-2)
    print(net['s1'].output_shape)  
    net['out'] = lasagne.layers.SliceLayer(net['s1'], slice(0, 80), axis=-1)
    print(net['out'] .output_shape)
    
    return net['out']




