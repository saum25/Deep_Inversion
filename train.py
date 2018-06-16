#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Created on 22 Aug 2017

Code to train an upconvolutional neural network for
inverting a feature representation captured by Jan's
SVD [ISMIR 2015].
@author: Saumitra
'''

import argparse
import os
import io
from progress import progress
from simplecache import cached
import audio
import numpy as np
from labels import create_aligned_targets
import theano
import theano.tensor as T
floatX = theano.config.floatX
import lasagne
import znorm
import augment
import model
import upconv
import sys

# String formatting in python %.3f does rounding so need to write a separate function
def truncate(loss):
    return (int(loss * 100)/float(100))

def cal_excerpts(excerpts_indices):
    n_excerpt = 0
    
    for idx in excerpts_indices:
        n_excerpt+=len(idx)
    
    return n_excerpt

def argument_parser():
    parser = argparse.ArgumentParser(description='Trains an upconvolutional neural network for feature inversion')
    # Positional arguments
    parser.add_argument('prediction_file', action='store', help='file to load the prediction network weights from (.npz format)')
    parser.add_argument('generator_file', action='store', help='file to save the generator network weights to (.npz format)')
    # Optional arguments
    parser.add_argument('--dataset', default='jamendo', help='Name of the dataset to use.')
    parser.add_argument('--cache-spectra', metavar='DIR', default=None, help='Store spectra in the given directory (disabled by default).')
    parser.add_argument('--augment', action='store_true', default=True, help='If given, perform train-time data augmentation.')
    parser.add_argument('--no-augment', action='store_false', dest='augment', help='If given, disable train-time data augmentation.')
    parser.add_argument('--n_conv_layers', default=1, type=int, help='number of 3x3 conv layers to be added before upconv layers')
    parser.add_argument('--n_conv_filters', default=32, type=int, help='number of filters per conv layer in the upconvolutonal architecture for Conv layer inversion')
    parser.add_argument('--w_inputloss', default=1.0, type=float, help='input space loss needs to be scaled by this factor')
    parser.add_argument('--comp_layer_name', default='fc8', help='layer from which the comparator features are extracted')
    parser.add_argument('--nofeatloss', default=True, action='store_false', help='If given do not train with feature space loss')
    
    # lr decay schedule
    parser.add_argument('--lr_init', default= 0.001, type =float, help='initial learning rate')
    parser.add_argument('--lr_decay', default= 0, type =int, help=' performs learning rate decay based on the selected schedule.') 

    return parser
    
def main():

    parser = argument_parser()
    args = parser.parse_args()

    # default arguments    
    sample_rate = 22050
    frame_len = 1024
    fps = 70
    mel_bands = 80
    mel_min = 27.5
    mel_max = 8000
    blocklen = 115
    batchsize = 32
    step = 10
    
    bin_nyquist = frame_len // 2 + 1
    bin_mel_max = bin_nyquist * 2 * mel_max // sample_rate
    
    # prepare dataset
    datadir = os.path.join(os.path.dirname(__file__), os.path.pardir, 'datasets', args.dataset)

    # load filelist - training and vaildation
    with io.open(os.path.join(datadir, 'filelists', 'train')) as f:
        filelist_tr = [l.rstrip() for l in f if l.rstrip()]
        
    with io.open(os.path.join(datadir, 'filelists', 'valid')) as f:
        filelist_va = [l.rstrip() for l in f if l.rstrip()]

    # compute spectra
    print("Computing%s spectra..." %
          (" or loading" if args.cache_spectra else ""))
    spects_tr = []
    spects_va = []
    
    print("\r====Training Files====")
    for fn in progress(filelist_tr, 'File '):
        cache_fn = (args.cache_spectra and os.path.join(args.cache_spectra, fn + '.npy'))
        spects_tr.append(cached(cache_fn, audio.extract_spect, os.path.join(datadir, 'audio', fn),sample_rate, frame_len, fps))
 
    print("\r====Validation Files====")
    for fn in progress(filelist_va, 'File '):
        cache_fn = (args.cache_spectra and os.path.join(args.cache_spectra, fn + '.npy'))
        spects_va.append(cached(cache_fn, audio.extract_spect, os.path.join(datadir, 'audio', fn),sample_rate, frame_len, fps))

    
    # load and convert corresponding labels
    print("Loading labels...")

    labels_tr = []
    labels_va = []
    for fn, spect in zip(filelist_tr, spects_tr):
        fn = os.path.join(datadir, 'labels', fn.rsplit('.', 1)[0] + '.lab')
        with io.open(fn) as f:
            segments = [l.rstrip().split() for l in f if l.rstrip()]
        segments = [(float(start), float(end), label == 'sing')
                    for start, end, label in segments]
        timestamps = np.arange(len(spect)) / float(fps)
        labels_tr.append(create_aligned_targets(segments, timestamps, np.bool))
        
    for fn, spect in zip(filelist_va, spects_va):
        fn = os.path.join(datadir, 'labels', fn.rsplit('.', 1)[0] + '.lab')
        with io.open(fn) as f:
            segments = [l.rstrip().split() for l in f if l.rstrip()]
        segments = [(float(start), float(end), label == 'sing')
                    for start, end, label in segments]
        timestamps = np.arange(len(spect)) / float(fps)
        labels_va.append(create_aligned_targets(segments, timestamps, np.bool))

    
    
    # prepare mel filterbank
    filterbank = audio.create_mel_filterbank(sample_rate, frame_len, mel_bands,
                                             mel_min, mel_max)
    filterbank = filterbank[:bin_mel_max].astype(floatX)    
    
    # precompute mel spectra, if needed, otherwise just define a generator
    mel_spects_tr = (np.log(np.maximum(np.dot(spect[:, :bin_mel_max], filterbank),1e-7)) for spect in spects_tr) # it is not clear why is Jan using natural log in place of log 10.
    mel_spects_va = (np.log(np.maximum(np.dot(spect[:, :bin_mel_max], filterbank),1e-7)) for spect in spects_va) 
    
    if not args.augment:
        mel_spects_tr = list(mel_spects_tr)
        del spects_tr
        
        mel_spects_va = list(mel_spects_va)
        del spects_va
        
    # - load mean/std or compute it, if not computed yet
    meanstd_file = os.path.join(os.path.dirname(__file__), '%s_meanstd.npz' % args.dataset)
    try:
        with np.load(meanstd_file) as f:
            mean = f['mean']
            std = f['std']
    except (IOError, KeyError):
        print("Computing mean and standard deviation...")
        mean, std = znorm.compute_mean_std(mel_spects_tr)
        np.savez(meanstd_file, mean=mean, std=std)
        
    mean = mean.astype(floatX)
    istd = np.reciprocal(std).astype(floatX)
    
    # - prepare training data generator
    print("\rPreparing training data feed...")
    if not args.augment:
        # Without augmentation, we just precompute the normalized mel spectra
        # and create a generator that returns mini-batches of random excerpts
        mel_spects_tr = [(spect - mean) * istd for spect in mel_spects_tr]
        excerpt_indices_tr = augment.select_excerpt_indices(mel_spects_tr, blocklen, step)
        #batches_tr = augment.grab_random_excerpts(mel_spects_tr, labels_tr, batchsize, blocklen)
        
        mel_spects_va = [(spect - mean) * istd for spect in mel_spects_va]
        excerpt_indices_va = augment.select_excerpt_indices(mel_spects_va, blocklen, step)
        #batches_va = augment.grab_random_excerpts(mel_spects_va, labels_va, batchsize, blocklen)
    else:
        #####CAUTION###### THE DATA AUGMENT CODE WILL NOT WORK AUTOMATICALLY AS THE TRAINING AND VALIDATION SUPPORT IS NOT ADDED CLEANLY.
        # IT JUST SHOWS THE TRAINING VARIBALES
        # For time stretching and pitch shifting, it pays off to preapply the
        # spline filter to each input spectrogram, so it does not need to be
        # applied to each mini-batch later.
        spline_order = 2
        if spline_order > 1:
            from scipy.ndimage import spline_filter
            spects = [spline_filter(spect, spline_order).astype(floatX) for spect in spects_tr]
        
        # We define a function to create the mini-batch generator. This allows
        # us to easily create multiple generators for multithreading if needed.
        def create_datafeed(spects, labels):
            # With augmentation, as we want to apply random time-stretching,
            # we request longer excerpts than we finally need to return.
            max_stretch = .3
            batches = augment.grab_random_excerpts(
                    spects, labels, batchsize=batchsize,
                    frames=int(blocklen / (1 - max_stretch)))

            # We wrap the generator in another one that applies random time
            # stretching and pitch shifting, keeping a given number of frames
            # and bins only.
            max_shift = .3

            batches = augment.apply_random_stretch_shift(
                                                         batches, max_stretch, max_shift,
                                                         keep_frames=blocklen, keep_bins= bin_mel_max,
                                                         order=spline_order, prefiltered=True)

            # We transform the excerpts to mel frequency and log magnitude.
            batches = augment.apply_filterbank(batches, filterbank)
            batches = augment.apply_logarithm(batches)
            # We apply random frequency filters
            batches = augment.apply_random_filters(batches, filterbank,
                                               mel_max, max_db=10)

                
            # We apply normalization
            batches = augment.apply_znorm(batches, mean, istd)

            return batches
        
        # We start the mini-batch generator and augmenter in one or more
        # background threads or processes (unless disabled).
        bg_threads = 3
        bg_processes = 0
        if not bg_threads and not bg_processes:
            # no background processing: just create a single generator
            batches = create_datafeed(spects, labels_tr)
        elif bg_threads:
            # multithreading: create a separate generator per thread
            batches = augment.generate_in_background(
                    [create_datafeed(spects, labels_tr)
                     for _ in range(bg_threads)],
                    num_cached=bg_threads * 5)
        elif bg_processes:
            # multiprocessing: single generator is forked along with processes
            batches = augment.generate_in_background(
                    [create_datafeed(spects, labels_tr)] * bg_processes,
                    num_cached=bg_processes * 25,
                    in_processes=True)
    
    print("Preparing prediction and generator functions...")
    # we create two functions by using two network architectures. One uses the pre-trained network
    # the other trains an upconvolutional network.

    #>>>>>>>>>>>>>>>>>>
    # Encoder Network - Network 1
    # instantiate neural network : Using the pretrained network: SchluterNet
    input_var = T.tensor3('input')
    inputs = input_var.dimshuffle(0, 'x', 1, 2)  # insert "channels" dimension, changes a 32 x 115 x 80 input to 32 x 1 x 115 x 80 input which is fed to the CNN
    
    network = model.architecture(inputs, (None, 1, blocklen, mel_bands))
    
    # load saved weights
    with np.load(args.prediction_file) as f:
        lasagne.layers.set_all_param_values(
                network['fc9'], [f['param%d' % i] for i in range(len(f.files))])

    # create output expression
    outputs_score = lasagne.layers.get_output(network['mp3'], deterministic=True) # enc_layer_name refers to the SVDNet(encoder) layer that needs to be inverted.

    # prepare and compile prediction function
    print("Compiling prediction function...")
    pred_fn = theano.function([input_var], outputs_score, allow_input_downcast=True)
    print (lasagne.layers.get_output_shape(network['mp3']))   # change here for layer name
    #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # Comparator 
    if (args.nofeatloss == True):
        print("In Feature Space Loss Case!!")
        # create comparator output expression. As we need to analyse features from which layers are better
        comparator_output = lasagne.layers.get_output(network[args.comp_layer_name], deterministic=True) # change here for playing with a layer
        # prepare and compile prediction function
        print("Compiling comparator function...")
        comp_fn = theano.function([input_var], comparator_output, allow_input_downcast=True)
    
    # training the Upconvolutional network - Network 2
    
    #input_var_deconv = T.matrix('input_var_deconv') # extracted features from the real input : a matrix of size 32x64(fc8 layer)
    input_var_deconv = T.tensor4('input_var_deconv')

    if (args.nofeatloss == True):
        # variables to handle feature space loss
        if (args.comp_layer_name =='fc9') or (args.comp_layer_name =='fc8') or (args.comp_layer_name == 'fc7'):
            input_var_gen_feat = T.matrix('input_var_gen_feat') # extracted features from the reconstructed input
            input_var_comp_feat = T.matrix('input_var_comp_feat') # extracted features from the real input but may be at a different layer
        else:
            input_var_gen_feat = T.tensor4('input_var_gen_feat')
            input_var_comp_feat = T.tensor4('input_var_comp_feat')
    
    #inputs_deconv = input_var_deconv.dimshuffle(0, 1, 'x', 'x') # 32 x 64 x 1 x 1. Adding the width and depth dimensions
    #gen_network = upconv.architecture_upconv_fc7(input_var_deconv, (batchsize, lasagne.layers.get_output_shape(network['fc7'])[1])) # change here for fc8 vs fc7 inversion
    gen_network = upconv.architecture_upconv_c2_mp3(input_var_deconv, (batchsize, lasagne.layers.get_output_shape(network['mp3'])[1], lasagne.layers.get_output_shape(network['mp3'])[2], lasagne.layers.get_output_shape(network['mp3'])[3]), args.n_conv_layers, args.n_conv_filters)
    outputs = lasagne.layers.get_output(gen_network, deterministic=False)
    
    gen_fn = theano.function([input_var_deconv], outputs, allow_input_downcast= True)   # takes in features and gives out reconstructed output
    
    # create cost expression
    # loss: squared euclidean distance per sample in a batch
    input_space_loss = T.mean(lasagne.objectives.squared_error(outputs, inputs))
    # feature space loss: L2 loss between the features extracted from inverted input and actual features
    if (args.nofeatloss == True):
        feat_space_loss = T.mean(lasagne.objectives.squared_error(input_var_comp_feat, input_var_gen_feat))
    # add weight decay or regularisation loss for training: needs to be checked as it appears to be regularising all layers.
    all_layers = lasagne.layers.get_all_layers(gen_network)
    l2_penalty = lasagne.regularization.regularize_layer_params(all_layers, lasagne.regularization.l2) * 0.0001
    if (args.nofeatloss == True):
        print("comes here!!!!")
        cost = input_space_loss * args.w_inputloss + feat_space_loss + l2_penalty
    else:
        cost = input_space_loss + l2_penalty
        
    # prepare and compile training function
    params = lasagne.layers.get_all_params(gen_network, trainable=True)
    #momentum = 0.95
    
    # learning rate params
    initial_eta = args.lr_init
    # decay by 10% in each epoch
    eta_decay_fix = 0.1
    # decay by 50% on demand
    eta_decay_variable = 0.5
    # flags to control decay schedule
    decay_schedule = args.lr_decay


    loss_prev_epoch = 0
    loss_current_epoch = 0
    
    eta = theano.shared(lasagne.utils.floatX(initial_eta))
    #updates = lasagne.updates.nesterov_momentum(cost, params, eta, momentum)
    updates = lasagne.updates.adam(cost, params, eta)
    print("Compiling training function...")
    if (args.nofeatloss == True):
        train_fn = theano.function([input_var_deconv, input_var_comp_feat, input_var, input_var_gen_feat], cost, updates=updates, allow_input_downcast=True)
    else:    
        train_fn = theano.function([input_var_deconv, input_var], cost, updates=updates, allow_input_downcast=True)
        
    print("Compiling validation function...")
    
    outputs_val = lasagne.layers.get_output(gen_network, deterministic=True)
    cost_val = T.mean(lasagne.objectives.squared_error(outputs_val, inputs))    
    val_fn = theano.function([input_var_deconv, input_var], cost_val, allow_input_downcast=True)
    
    # run the training loop
    print("\rTraining the inversion network:")
    num_excerpts_tr = cal_excerpts(excerpt_indices_tr)
    num_excerpts_va = cal_excerpts(excerpt_indices_va)

    epochs = 30
    epochsize_tr = num_excerpts_tr/batchsize
    epochsize_va = num_excerpts_va/batchsize

    # remove the extra indices from the last file
    excerpt_indices_tr[-1] = excerpt_indices_tr[-1][:-(num_excerpts_tr%batchsize)]
    excerpt_indices_va[-1] = excerpt_indices_va[-1][:-(num_excerpts_va%batchsize)]
    
    batches_tr = augment.grab_random_excerpts(mel_spects_tr, labels_tr, batchsize, blocklen, excerpt_indices_tr)
    batches_va = augment.grab_random_excerpts(mel_spects_va, labels_va, batchsize, blocklen, excerpt_indices_va)

    batches_tr = iter(batches_tr)
    batches_va = iter(batches_va)
    
    patience = 15
    
    list_training_log = []
    list_training_log_save = [] # looks redundant, but no time to fix it
    
    for epoch in range(epochs):
        print("\rLearning rate epoch %d: %f" %(epoch, eta.get_value()))
        err = 0
        for batch_tr in progress(
                range(epochsize_tr), min_delay=.5,
                desc='Epoch %d/%d: Batch ' % (epoch + 1, epochs)):
            data, labels = next(batches_tr) # followed a simple styple *next(batches) from Jan is unclear what is passed.
            # labels information is a dummy variable its not used in training.
            pred = pred_fn(data)    # a theano function returns a numpy array always. Here it is a matrix of shape 32  x 64
            if (args.nofeatloss == True):
                comp_feat = comp_fn(data)
                gen_output = np.squeeze(gen_fn(pred), axis=1) # output shape: 32 x 1 x 115 x 80
                gen_output_feat = comp_fn(gen_output) # output shape: 32 x 64
                err += train_fn(pred, comp_feat, data, gen_output_feat)
            else:
                err += train_fn(pred, data)

            if not np.isfinite(err):
                print("\nEncountered NaN loss in training. Aborting.")
                sys.exit(1)
                
        if epoch + 1 == 1 or epoch + 1 == epochs: # save training error for the first epoch or the last epoch
            list_training_log.append(err/epochsize_tr)
        print("Train loss: %.3f" % (err / epochsize_tr))
        loss_current_epoch = err/ epochsize_tr
        
        # Running the evaluation on training set after training one epoch, just to see how low is the error
        '''err = 0
        for batch_tr in progress(
        range(epochsize_tr), min_delay=.5,
        desc='Epoch %d/%d: Batch ' % (epoch + 1, epochs)):
            data, labels = next(batches_tr) # followed a simple styple *next(batches) from Jan is unclear what is passed.
        # labels information is a dummy variable its not used in training.
        pred = pred_fn(data)    # a theano function returns a numpy array always. Here it is a matrix of shape 32  x 64
        err += val_fn(pred, data)

        if not np.isfinite(err):
            print("\nEncountered NaN loss in training. Aborting.")
            sys.exit(1)
        print(" Testing the train data :Train loss: %.3f" % (err / epochsize_tr))'''
    
        # Calculating validation loss per epoch
        err_va = 0
        for batch_va in progress(
                range(epochsize_va), min_delay=.5,
                desc='Epoch %d/%d: Batch ' % (epoch + 1, epochs)):
            data, labels = next(batches_va) # followed a simple styple *next(batches) from Jan is unclear what is passed.
            # labels information is a dummy variable its not used in training.
            pred = pred_fn(data)    # a theano function returns a numpy array always. Here it is a matrix of shape 32  x 64
            err_va += val_fn(pred, data)
            
            if not np.isfinite(err):
                print("\nEncountered NaN loss in training. Aborting.")
                sys.exit(1)
        print("Validation loss: %.3f" % (err_va / epochsize_va))
        
        # learning rate decay
        # decaying the learning rate by different schemes
        print("\rTraining loss: previous epoch: %f current epoch:%f" %(loss_prev_epoch, loss_current_epoch))
        
        if decay_schedule==1:
            print("Decaying lr, based on schedule 1")
            eta.set_value(eta.get_value() * lasagne.utils.floatX(eta_decay_fix))
        elif decay_schedule==2:
            if epoch == 15 or epoch == 22:
                print("Decaying lr, based on schedule 2")
                eta.set_value(eta.get_value() * lasagne.utils.floatX(eta_decay_variable))
        elif decay_schedule==3:
            if truncate(loss_prev_epoch) == truncate(loss_current_epoch):
                print("Decaying lr, based on schedule 3")
                eta.set_value(eta.get_value() * lasagne.utils.floatX(eta_decay_variable))
        else:
            print("Decaying lr, based on schedule 0") # don't decay the learning rate at all
            
        loss_prev_epoch = loss_current_epoch

        # code for early stopping the model training to prevent overfitting   
             
        '''if epoch == 0:# save the model parameters after first epoch
            print("----------------------------------Saving model after epoch:%d" %(epoch+1))
            np.savez(args.generator_file, **{'param%d' % i: p for i, p in enumerate(lasagne.layers.get_all_param_values(gen_network))})
            val_loss = err_va / epochsize_va
            train_loss = loss_current_epoch
        else: # update the model iff training and validation both losses are coming down
            if (truncate(err_va / epochsize_va) <= truncate(val_loss)) and (truncate(loss_current_epoch) - truncate(train_loss) < 0):
                print("-------------------------------Saving model after epoch:%d" %(epoch+1))
                np.savez(args.generator_file, **{'param%d' % i: p for i, p in enumerate(lasagne.layers.get_all_param_values(gen_network))})
                val_loss = err_va / epochsize_va
                train_loss = loss_current_epoch
                patience = 15   # restarts from once the model is saved
            else:# else wait till patience runs down.
                if patience!=0:
                    print("waiting for the validation loss to decrease!!")
                    patience = patience - 1
                    print("Epochs left to view the loss decrease: %d" %(patience))
                else:
                    print("patience ran out......")
                    break'''                                
        
    # save final network
    print("Saving final model")
    np.savez(args.generator_file, **{'param%d' % i: p for i, p in enumerate(
            lasagne.layers.get_all_param_values(gen_network))})
    
    # saving the training logs
    # information about the start training loss, end training loss and the change/improvement in loss
    list_training_log.append(list_training_log[0]-list_training_log[1])
    list_training_log_save.append(tuple(list_training_log))
    with open('models/mp3/training_log.txt', 'a+') as fp:
        fp.write('\n'.join('{} {} {}'.format(x[0],x[1],x[2]) for x in list_training_log_save))
        fp.write('\n')
            


if __name__ == '__main__':
    main()
