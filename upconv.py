'''
Created on 26 Aug 2017

@author: Saumitra Mishra
'''

import lasagne
from lasagne.layers import (InputLayer, DenseLayer, ReshapeLayer, TransposedConv2DLayer, batch_norm, Conv2DLayer)

def architecture_upconv_fc9(input_var, input_shape):
    
    net = {}
    #number of filters in the uconv layer
    n_filters = 64
    
    net['data'] = InputLayer(input_shape, input_var)
    print("\n")
    print("Input data shape")
    print(net['data'].output_shape)
    print("Layer-wise output shape")
    net['fc1'] = batch_norm(DenseLayer(net['data'], num_units=64, W=lasagne.init.HeNormal(), nonlinearity=lasagne.nonlinearities.elu))
    print(net['fc1'].output_shape)
    net['fc2'] = batch_norm(DenseLayer(net['fc1'], num_units=64, W=lasagne.init.HeNormal(), nonlinearity=lasagne.nonlinearities.elu))
    print(net['fc2'].output_shape)
    net['fc3'] = batch_norm(DenseLayer(net['fc2'], num_units=256, W=lasagne.init.HeNormal(), nonlinearity=lasagne.nonlinearities.elu))
    print(net['fc3'].output_shape)
    net['rs1'] = ReshapeLayer(net['fc3'], (32, 16, 4, 4)) # assuming that the shape is batch x depth x row x columns
    
    kwargs = dict(nonlinearity=lasagne.nonlinearities.elu,
                  W=lasagne.init.HeNormal())
    
    net['uc1'] = batch_norm(TransposedConv2DLayer(net['rs1'], num_filters= n_filters, filter_size= 4, stride = 2, crop=1, **kwargs))
    print(net['uc1'].output_shape)
    net['c1'] = batch_norm(Conv2DLayer(net['uc1'], num_filters= n_filters, filter_size= 3, stride = 1, pad=1, **kwargs))
    print(net['c1'].output_shape)    
    
    net['uc2'] = batch_norm(TransposedConv2DLayer(net['c1'], num_filters= n_filters/2, filter_size= 4, stride = 2, crop=1, **kwargs))
    print(net['uc2'].output_shape)
    net['c2'] = batch_norm(Conv2DLayer(net['uc2'], num_filters= n_filters/2, filter_size= 3, stride = 1, pad=1, **kwargs))
    print(net['c2'].output_shape)    
    
    net['uc3'] = batch_norm(TransposedConv2DLayer(net['c2'], num_filters= n_filters/4, filter_size= 4, stride = 2, crop=1, **kwargs))
    print(net['uc3'].output_shape)
    net['c3'] = batch_norm(Conv2DLayer(net['uc3'], num_filters= n_filters/4, filter_size= 3, stride = 1, pad=1, **kwargs))
    print(net['c3'].output_shape)

    net['uc4'] = batch_norm(TransposedConv2DLayer(net['c3'], num_filters= n_filters/8, filter_size= 4, stride = 2, crop=1, **kwargs))
    print(net['uc4'].output_shape)
    net['c4'] = batch_norm(Conv2DLayer(net['uc4'], num_filters= n_filters/8, filter_size= 3, stride = 1, pad=1, **kwargs))
    print(net['c4'].output_shape)
    
    net['uc5'] = TransposedConv2DLayer(net['c4'], num_filters= 1, filter_size= 4, stride = 2, crop=1, **kwargs)
    print(net['uc5'].output_shape)
    '''net['uc5'] = TransposedConv2DLayer(net['c4'], num_filters= n_filters/16, filter_size= 4, stride = 2, crop=1, **kwargs)
    print(net['uc5'].output_shape)
    net['c5'] = batch_norm(Conv2DLayer(net['uc5'], num_filters= n_filters/16, filter_size= 3, stride = 1, pad=1, **kwargs))
    print(net['c5'].output_shape)
    
    net['uc6'] = TransposedConv2DLayer(net['c5'], num_filters= 1, filter_size= 4, stride = 2, crop=1, **kwargs)
    print(net['uc6'].output_shape)'''
    

    # slicing the output to 115 x 80 size
    net['s1'] = lasagne.layers.SliceLayer(net['uc5'], slice(0, 115), axis=-2)
    net['out'] = lasagne.layers.SliceLayer(net['s1'], slice(0, 80), axis=-1)
    print(net['out'] .output_shape)
    
    print("Number of parameter to be learned: %d\n" %(lasagne.layers.count_params(net['out'])))
    
    return net['out']

def architecture_upconv_fc8(input_var, input_shape):
    
    net = {}
    #number of filters in the uconv layer
    n_filters = 64
    
    net['data'] = InputLayer(input_shape, input_var)
    print("\n")
    print("Input data shape")
    print(net['data'].output_shape)
    print("Layer-wise output shape")
    net['fc1'] = batch_norm(DenseLayer(net['data'], num_units=64, W=lasagne.init.HeNormal(), nonlinearity=lasagne.nonlinearities.elu))
    print(net['fc1'].output_shape)
    net['fc2'] = batch_norm(DenseLayer(net['fc1'], num_units=256, W=lasagne.init.HeNormal(), nonlinearity=lasagne.nonlinearities.elu))
    print(net['fc2'].output_shape)
    net['rs1'] = ReshapeLayer(net['fc2'], (32, 16, 4, 4)) # assuming that the shape is batch x depth x row x columns
    
    kwargs = dict(nonlinearity=lasagne.nonlinearities.elu,
                  W=lasagne.init.HeNormal())
    
    net['uc1'] = batch_norm(TransposedConv2DLayer(net['rs1'], num_filters= n_filters, filter_size= 4, stride = 2, crop=1, **kwargs))
    print(net['uc1'].output_shape)
    net['c1'] = batch_norm(Conv2DLayer(net['uc1'], num_filters= n_filters, filter_size= 3, stride = 1, pad=1, **kwargs))
    print(net['c1'].output_shape)    
    
    net['uc2'] = batch_norm(TransposedConv2DLayer(net['c1'], num_filters= n_filters/2, filter_size= 4, stride = 2, crop=1, **kwargs))
    print(net['uc2'].output_shape)
    net['c2'] = batch_norm(Conv2DLayer(net['uc2'], num_filters= n_filters/2, filter_size= 3, stride = 1, pad=1, **kwargs))
    print(net['c2'].output_shape)    
    
    net['uc3'] = batch_norm(TransposedConv2DLayer(net['c2'], num_filters= n_filters/4, filter_size= 4, stride = 2, crop=1, **kwargs))
    print(net['uc3'].output_shape)
    net['c3'] = batch_norm(Conv2DLayer(net['uc3'], num_filters= n_filters/4, filter_size= 3, stride = 1, pad=1, **kwargs))
    print(net['c3'].output_shape)

    net['uc4'] = batch_norm(TransposedConv2DLayer(net['c3'], num_filters= n_filters/8, filter_size= 4, stride = 2, crop=1, **kwargs))
    print(net['uc4'].output_shape)
    net['c4'] = batch_norm(Conv2DLayer(net['uc4'], num_filters= n_filters/8, filter_size= 3, stride = 1, pad=1, **kwargs))
    print(net['c4'].output_shape)
    
    net['uc5'] = TransposedConv2DLayer(net['c4'], num_filters= 1, filter_size= 4, stride = 2, crop=1, **kwargs)
    print(net['uc5'].output_shape)
    '''net['uc5'] = TransposedConv2DLayer(net['c4'], num_filters= n_filters/16, filter_size= 4, stride = 2, crop=1, **kwargs)
    print(net['uc5'].output_shape)
    net['c5'] = batch_norm(Conv2DLayer(net['uc5'], num_filters= n_filters/16, filter_size= 3, stride = 1, pad=1, **kwargs))
    print(net['c5'].output_shape)
    
    net['uc6'] = TransposedConv2DLayer(net['c5'], num_filters= 1, filter_size= 4, stride = 2, crop=1, **kwargs)
    print(net['uc6'].output_shape)'''
    

    # slicing the output to 115 x 80 size
    net['s1'] = lasagne.layers.SliceLayer(net['uc5'], slice(0, 115), axis=-2)
    net['out'] = lasagne.layers.SliceLayer(net['s1'], slice(0, 80), axis=-1)
    print(net['out'] .output_shape)
    
    print("Number of parameter to be learned: %d\n" %(lasagne.layers.count_params(net['out'])))
    
    return net['out']

def architecture_upconv_fc7(input_var, input_shape):
    
    net = {}
    n_filters = 64
    
    net['data'] = InputLayer(input_shape, input_var)
    print(net['data'].output_shape)
    net['fc2'] = batch_norm(DenseLayer(net['data'], num_units=256, W=lasagne.init.HeNormal(), nonlinearity=lasagne.nonlinearities.elu))
    print(net['fc2'].output_shape)
    
    net['rs1'] = ReshapeLayer(net['fc2'], (32, 16, 4, 4)) # assuming that the shape is batch x depth x row x columns
    
    kwargs = dict(nonlinearity=lasagne.nonlinearities.elu,
                  W=lasagne.init.HeNormal())
    
    net['uc1'] = batch_norm(TransposedConv2DLayer(net['rs1'], num_filters= n_filters, filter_size= 4, stride = 2, crop=1, **kwargs))
    print(net['uc1'].output_shape)
    net['c1'] = batch_norm(Conv2DLayer(net['uc1'], num_filters= n_filters, filter_size= 3, stride = 1, pad=1, **kwargs))
    print(net['c1'].output_shape)    
    
    net['uc2'] = batch_norm(TransposedConv2DLayer(net['c1'], num_filters= n_filters/2, filter_size= 4, stride = 2, crop=1, **kwargs))
    print(net['uc2'].output_shape)
    net['c2'] = batch_norm(Conv2DLayer(net['uc2'], num_filters= n_filters/2, filter_size= 3, stride = 1, pad=1, **kwargs))
    print(net['c2'].output_shape)    
    
    net['uc3'] = batch_norm(TransposedConv2DLayer(net['c2'], num_filters= n_filters/4, filter_size= 4, stride = 2, crop=1, **kwargs))
    print(net['uc3'].output_shape)
    net['c3'] = batch_norm(Conv2DLayer(net['uc3'], num_filters= n_filters/4, filter_size= 3, stride = 1, pad=1, **kwargs))
    print(net['c3'].output_shape)

    net['uc4'] = batch_norm(TransposedConv2DLayer(net['c3'], num_filters= n_filters/8, filter_size= 4, stride = 2, crop=1, **kwargs))
    print(net['uc4'].output_shape)
    net['c4'] = batch_norm(Conv2DLayer(net['uc4'], num_filters= n_filters/8, filter_size= 3, stride = 1, pad=1, **kwargs))
    print(net['c4'].output_shape)
    
    net['uc5'] = TransposedConv2DLayer(net['c4'], num_filters= 1, filter_size= 4, stride = 2, crop=1, **kwargs)
    print(net['uc5'].output_shape)

    # slicing the output to 115 x 80 size
    net['s1'] = lasagne.layers.SliceLayer(net['uc5'], slice(0, 115), axis=-2)
    net['out'] = lasagne.layers.SliceLayer(net['s1'], slice(0, 80), axis=-1)
    print(net['out'] .output_shape)
    
    print("\r Number of parameter to be learned: %d" %(lasagne.layers.count_params(net['out'])))
    
    return net['out']

def architecture_upconv_c5_mp6(input_var, input_shape, n_conv_layers, n_conv_filters):
    
    net = {}
    
    kwargs = dict(nonlinearity=lasagne.nonlinearities.elu,
                  W=lasagne.init.HeNormal())
    
    net['data'] = InputLayer(input_shape, input_var)
    print(net['data'].output_shape)
    print("\r Layer-by-layer output shapes of the upconvolutional network")
    
    # Bunch of 3 x 3 convolution layers: experimentally we found that, adding 3 conv layers in start than in middle is better: but why?
    net['c1'] = batch_norm(Conv2DLayer(net['data'], num_filters= 64, filter_size= 3, stride = 1, pad=1, **kwargs))
    print(net['c1'].output_shape)   
    net['c2'] = batch_norm(Conv2DLayer(net['c1'], num_filters= 64, filter_size= 3, stride = 1, pad=1, **kwargs))
    print(net['c2'].output_shape)
    net['c3'] = batch_norm(Conv2DLayer(net['c2'], num_filters= 64, filter_size= 3, stride = 1, pad=1, **kwargs))
    print(net['c3'].output_shape)
    net['c4'] = batch_norm(Conv2DLayer(net['c3'], num_filters= 64, filter_size= 3, stride = 1, pad=1, **kwargs))
    print(net['c4'].output_shape)
    '''net['c5'] = batch_norm(Conv2DLayer(net['c4'], num_filters= 64, filter_size= 3, stride = 1, pad=1, **kwargs))
    print(net['c5'].output_shape)'''
    
    # Bunch of transposed convolution layers  
    net['uc1'] = batch_norm(TransposedConv2DLayer(net['c4'], num_filters= 32, filter_size= 4, stride = 2, crop=1, **kwargs))
    print(net['uc1'].output_shape)
    '''net['c1'] = batch_norm(Conv2DLayer(net['uc1'], num_filters= 32, filter_size= 3, stride = 1, pad=1, **kwargs))
    print(net['c1'].output_shape)'''
    
    net['uc2'] = batch_norm(TransposedConv2DLayer(net['uc1'], num_filters= 16, filter_size= 4, stride = 2, crop=1, **kwargs))
    print(net['uc2'].output_shape)
    '''net['c2'] = batch_norm(Conv2DLayer(net['uc2'], num_filters= 16, filter_size= 3, stride = 1, pad=1, **kwargs))
    print(net['c2'].output_shape)''' 
    
    net['uc3'] = batch_norm(TransposedConv2DLayer(net['uc2'], num_filters= 8, filter_size= 4, stride = 2, crop=1, **kwargs))
    print(net['uc3'].output_shape)
    '''net['c3'] = batch_norm(Conv2DLayer(net['uc3'], num_filters= 8, filter_size= 3, stride = 1, pad=1, **kwargs))
    print(net['c3'].output_shape)'''

    net['uc4'] = batch_norm(TransposedConv2DLayer(net['uc3'], num_filters= 1, filter_size= 4, stride = 2, crop=1, **kwargs))
    print(net['uc4'].output_shape)

    # slicing the output to 115 x 80 size  
    net['s1'] = lasagne.layers.SliceLayer(net['uc4'], slice(0, 115), axis=-2)
    print(net['s1'].output_shape)  
    net['out'] = lasagne.layers.SliceLayer(net['s1'], slice(0, 80), axis=-1)
    print(net['out'] .output_shape)
    print("Number of parameter to be learned: %d" %(lasagne.layers.count_params(net['out'])))
    
    return net['out']

def architecture_upconv_conv5(input_var, input_shape, n_conv_layers, n_conv_filters):
    
    net = {}
    
    kwargs = dict(nonlinearity=lasagne.nonlinearities.elu,
                  W=lasagne.init.HeNormal())
    
    net['data'] = InputLayer(input_shape, input_var)
    print("\rLayer output shapes")
    print(net['data'].output_shape)
    
    # Bunch of 3 x 3 convolution layers: experimentally we found that, adding 3 conv layers in start than in middle is better: but why?
    '''net['c1'] = batch_norm(Conv2DLayer(net['data'], num_filters= 32, filter_size= 3, stride = 1, pad=1, **kwargs))
    print(net['c1'].output_shape)   
    net['c2'] = batch_norm(Conv2DLayer(net['c1'], num_filters= 32, filter_size= 3, stride = 1, pad=1, **kwargs))
    print(net['c2'].output_shape)
    net['c3'] = batch_norm(Conv2DLayer(net['c2'], num_filters= 32, filter_size= 3, stride = 1, pad=1, **kwargs))
    print(net['c3'].output_shape)'''
    
    i = 'data'
    j = 'c1'
    for idx in range(n_conv_layers):
        print("Conv layer index: %d" %(idx+1))
        net[j] = batch_norm(Conv2DLayer(net[i], num_filters= n_conv_filters, filter_size= 3, stride = 1, pad=1, **kwargs))
        print(net[j].output_shape)
        # renaming for next iteration
        i = j
        j = j[:-1] + str(idx + 2)        
    
    # Bunch of transposed convolution layers  
    net['uc1'] = batch_norm(TransposedConv2DLayer(net[i], num_filters= n_conv_filters/2, filter_size= 4, stride = 2, crop=1, **kwargs))
    print(net['uc1'].output_shape)
    
    net['uc2'] = batch_norm(TransposedConv2DLayer(net['uc1'], num_filters= 1, filter_size= 4, stride = 2, crop=1, **kwargs))
    print(net['uc2'].output_shape)

    # slicing the output to 115 x 80 size  
    net['s1'] = lasagne.layers.SliceLayer(net['uc2'], slice(0, 115), axis=-2)
    print(net['s1'].output_shape)  
    net['out'] = lasagne.layers.SliceLayer(net['s1'], slice(0, 80), axis=-1)
    print(net['out'] .output_shape)
    
    print("Number of parameter to be learned: %d" %(lasagne.layers.count_params(net['out'])))
    
    return net['out']


def architecture_upconv_conv4(input_var, input_shape, n_conv_layers, n_conv_filters):
    
    net = {}
    
    kwargs = dict(nonlinearity=lasagne.nonlinearities.elu,
                  W=lasagne.init.HeNormal())
    
    net['data'] = InputLayer(input_shape, input_var)
    print("\rLayer output shapes")
    print(net['data'].output_shape)
    
    # Bunch of 3 x 3 convolution layers: experimentally we found that, adding 3 conv layers in start than in middle is better: but why?
    '''net['c1'] = batch_norm(Conv2DLayer(net['data'], num_filters= 32, filter_size= 3, stride = 1, pad=1, **kwargs))
    print(net['c1'].output_shape)   
    net['c2'] = batch_norm(Conv2DLayer(net['c1'], num_filters= 32, filter_size= 3, stride = 1, pad=1, **kwargs))
    print(net['c2'].output_shape)
    net['c3'] = batch_norm(Conv2DLayer(net['c2'], num_filters= 32, filter_size= 3, stride = 1, pad=1, **kwargs))
    print(net['c3'].output_shape)'''
    
    i = 'data'
    j = 'c1'
    for idx in range(n_conv_layers):
        print("Conv layer index: %d" %(idx+1))
        net[j] = batch_norm(Conv2DLayer(net[i], num_filters= n_conv_filters, filter_size= 3, stride = 1, pad=1, **kwargs))
        print(net[j].output_shape)
        # renaming for next iteration
        i = j
        j = j[:-1] + str(idx + 2)        
    
    # Bunch of transposed convolution layers  
    net['uc1'] = batch_norm(TransposedConv2DLayer(net[i], num_filters= n_conv_filters/2, filter_size= 4, stride = 2, crop=1, **kwargs))
    print(net['uc1'].output_shape)
    
    net['uc2'] = batch_norm(TransposedConv2DLayer(net['uc1'], num_filters= 1, filter_size= 4, stride = 2, crop=1, **kwargs))
    print(net['uc2'].output_shape)

    # slicing the output to 115 x 80 size  
    net['s1'] = lasagne.layers.SliceLayer(net['uc2'], slice(0, 115), axis=-2)
    print(net['s1'].output_shape)  
    net['out'] = lasagne.layers.SliceLayer(net['s1'], slice(0, 80), axis=-1)
    print(net['out'] .output_shape)
    
    print("Number of parameter to be learned: %d" %(lasagne.layers.count_params(net['out'])))
    
    return net['out']

def architecture_upconv_c2_mp3(input_var, input_shape, n_conv_layers, n_conv_filters):
    
    net = {}
    
    kwargs = dict(nonlinearity=lasagne.nonlinearities.elu,
                  W=lasagne.init.HeNormal())
    
    net['data'] = InputLayer(input_shape, input_var)
    print("\rLayer output shapes")
    print(net['data'].output_shape)
    
    # Bunch of 3 x 3 convolution layers: experimentally we found that, adding 3 conv layers in start than in middle is better: but why?    
    i = 'data'
    j = 'c1'
    for idx in range(n_conv_layers):
        print("Conv layer index: %d" %(idx+1))
        net[j] = batch_norm(Conv2DLayer(net[i], num_filters= n_conv_filters, filter_size= 3, stride = 1, pad=1, **kwargs))
        print(net[j].output_shape)
        # renaming for next iteration
        i = j
        j = j[:-1] + str(idx + 2)        
    
    # Bunch of transposed convolution layers  
    net['uc1'] = batch_norm(TransposedConv2DLayer(net[i], num_filters= n_conv_filters/2, filter_size= 4, stride = 2, crop=1, **kwargs))
    print(net['uc1'].output_shape)
    
    net['uc2'] = batch_norm(TransposedConv2DLayer(net['uc1'], num_filters= 1, filter_size= 4, stride = 2, crop=1, **kwargs))
    print(net['uc2'].output_shape)

    # slicing the output to 115 x 80 size  
    net['s1'] = lasagne.layers.SliceLayer(net['uc2'], slice(0, 115), axis=-2)
    print(net['s1'].output_shape)  
    net['out'] = lasagne.layers.SliceLayer(net['s1'], slice(0, 80), axis=-1)
    print(net['out'] .output_shape)
    
    print("Number of parameter to be learned: %d" %(lasagne.layers.count_params(net['out'])))
    
    return net['out']


def architecture_upconv_c1(input_var, input_shape, n_conv_layers, n_conv_filters):
    
    net = {}
    
    kwargs = dict(nonlinearity=lasagne.nonlinearities.elu,
                  W=lasagne.init.HeNormal())
    
    net['data'] = InputLayer(input_shape, input_var)
    print("\rLayer output shapes")
    print(net['data'].output_shape)
    
    # Bunch of 3 x 3 convolution layers: experimentally we found that, adding 3 conv layers in start than in middle is better: but why?    
    i = 'data'
    j = 'c1'
    for idx in range(n_conv_layers):
        print("Conv layer index: %d" %(idx+1))
        net[j] = batch_norm(Conv2DLayer(net[i], num_filters= n_conv_filters, filter_size= 3, stride = 1, pad=1, **kwargs))
        print(net[j].output_shape)
        # renaming for next iteration
        i = j
        j = j[:-1] + str(idx + 2)     
    
    # Bunch of transposed convolution layers  
    net['uc1'] = batch_norm(TransposedConv2DLayer(net[i], num_filters= n_conv_filters/2, filter_size= 4, stride = 2, crop=1, **kwargs))
    print(net['uc1'].output_shape)
    
    net['uc2'] = batch_norm(TransposedConv2DLayer(net['uc1'], num_filters= 1, filter_size= 4, stride = 2, crop=1, **kwargs))
    print(net['uc2'].output_shape)

    # slicing the output to 115 x 80 size  
    net['s1'] = lasagne.layers.SliceLayer(net['uc2'], slice(0, 115), axis=-2)
    print(net['s1'].output_shape)  
    net['out'] = lasagne.layers.SliceLayer(net['s1'], slice(0, 80), axis=-1)
    print(net['out'] .output_shape)
    
    print("Number of parameter to be learned: %d" %(lasagne.layers.count_params(net['out'])))
    
    return net['out']




