#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Created on 22 Aug 2017

Code to train an upconvolutional neural network for
inverting a feature representation captured by Jan's
SVD

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

def argument_parser():
    parser = argparse.ArgumentParser(description='Trains an upconvolutional network for feature inversion')
    parser.add_argument('modelfile', action='store', help='file to load the learned weights from (.npz format)')
    parser.add_argument('outfile', action='store', help='file to save the trained upconv network weights to (.npz format)')
    parser.add_argument('--dataset', default='jamendo', help='Name of the dataset to use.')
    parser.add_argument('--cache-spectra', metavar='DIR', default=None, help='Store spectra in the given directory (disabled by default).')
    parser.add_argument('--augment', action='store_true', default=True, help='If given, perform train-time data augmentation.')
    parser.add_argument('--no-augment', action='store_false', dest='augment', help='If given, disable train-time data augmentation.') 

    return parser
    
def main():
    parser = argument_parser()
    args = parser.parse_args()
    
    sample_rate = 22050
    frame_len = 1024
    fps = 70
    mel_bands = 80
    mel_min = 27.5
    mel_max = 8000
    blocklen = 115
    batchsize = 32
    
    bin_nyquist = frame_len // 2 + 1
    bin_mel_max = bin_nyquist * 2 * mel_max // sample_rate
    
    # prepare dataset
    datadir = os.path.join(os.path.dirname(__file__), os.path.pardir, 'datasets', args.dataset)

    # load filelist
    with io.open(os.path.join(datadir, 'filelists', 'train')) as f:
        filelist = [l.rstrip() for l in f if l.rstrip()]

    # compute spectra
    print("Computing%s spectra..." %
          (" or loading" if args.cache_spectra else ""))
    spects = []
    for fn in progress(filelist, 'File '):
        cache_fn = (args.cache_spectra and os.path.join(args.cache_spectra, fn + '.npy'))
        spects.append(cached(cache_fn, audio.extract_spect, os.path.join(datadir, 'audio', fn),sample_rate, frame_len, fps))
    
    # load and convert corresponding labels
    print("Loading labels...")

    labels = []
    for fn, spect in zip(filelist, spects):
        fn = os.path.join(datadir, 'labels', fn.rsplit('.', 1)[0] + '.lab')
        with io.open(fn) as f:
            segments = [l.rstrip().split() for l in f if l.rstrip()]
        segments = [(float(start), float(end), label == 'sing')
                    for start, end, label in segments]
        timestamps = np.arange(len(spect)) / float(fps)
        labels.append(create_aligned_targets(segments, timestamps, np.bool))
        
    # prepare mel filterbank
    filterbank = audio.create_mel_filterbank(sample_rate, frame_len, mel_bands,
                                             mel_min, mel_max)
    filterbank = filterbank[:bin_mel_max].astype(floatX)    
    
    # precompute mel spectra, if needed, otherwise just define a generator
    mel_spects = (np.log(np.maximum(np.dot(spect[:, :bin_mel_max], filterbank),1e-7)) for spect in spects) # it is not clear why is Jan using natural log in place of log 10.
    
    if not args.augment:
        mel_spects = list(mel_spects)
        del spects
        
    # - load mean/std or compute it, if not computed yet
    meanstd_file = os.path.join(os.path.dirname(__file__), '%s_meanstd.npz' % args.dataset)
    try:
        with np.load(meanstd_file) as f:
            mean = f['mean']
            std = f['std']
    except (IOError, KeyError):
        print("Computing mean and standard deviation...")
        mean, std = znorm.compute_mean_std(mel_spects)
        np.savez(meanstd_file, mean=mean, std=std)
        
    mean = mean.astype(floatX)
    istd = np.reciprocal(std).astype(floatX)
    
    # - prepare training data generator
    print("Preparing training data feed...")
    if not args.augment:
        # Without augmentation, we just precompute the normalized mel spectra
        # and create a generator that returns mini-batches of random excerpts
        mel_spects = [(spect - mean) * istd for spect in mel_spects]
        batches = augment.grab_random_excerpts(mel_spects, labels, batchsize, blocklen)
    else:
        # For time stretching and pitch shifting, it pays off to preapply the
        # spline filter to each input spectrogram, so it does not need to be
        # applied to each mini-batch later.
        spline_order = 2
        if spline_order > 1:
            from scipy.ndimage import spline_filter
            spects = [spline_filter(spect, spline_order).astype(floatX) for spect in spects]
        
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
            batches = create_datafeed(spects, labels)
        elif bg_threads:
            # multithreading: create a separate generator per thread
            batches = augment.generate_in_background(
                    [create_datafeed(spects, labels)
                     for _ in range(bg_threads)],
                    num_cached=bg_threads * 5)
        elif bg_processes:
            # multiprocessing: single generator is forked along with processes
            batches = augment.generate_in_background(
                    [create_datafeed(spects, labels)] * bg_processes,
                    num_cached=bg_processes * 25,
                    in_processes=True)
    
    print("Preparing training functions...")
    # we create two functions by using two network architectures. One uses the pre-trained network
    # the other trains an upconvolutional network.
    
    # Prediction Network - Network 1
    # instantiate neural network : Using the pretrained network
    input_var = T.tensor3('input')
    inputs = input_var.dimshuffle(0, 'x', 1, 2)  # insert "channels" dimension, changes a 32 x 115 x 80 input to 32 x 1 x 115 x 80 input which is fed to the CNN
    
    network = model.architecture(inputs, (None, 1, blocklen, mel_bands))
    
    # load saved weights
    with np.load(args.modelfile) as f:
        lasagne.layers.set_all_param_values(
                network, [f['param%d' % i] for i in range(len(f.files))])

    # create output expression
    outputs_fc9 = lasagne.layers.get_output(network, deterministic=True)

    # prepare and compile prediction function
    print("Compiling prediction function...")
    pred_fn = theano.function([input_var], outputs_fc9)
    
    # training the Upconvolutional network - Network 2
    
    input_var_deconv = T.col('input_var_deconv')
    inputs_deconv = input_var_deconv.dimshuffle(0, 1, 'x', 'x') # 32x 1 x 1 x 1. Adding the width and depth dimensions
    net = upconv.architecture_upconv(inputs_deconv, (32, 1, 1, 1))
    
    # create cost expression
    outputs = lasagne.layers.get_output(net, deterministic=False)
    cost = T.mean(lasagne.objectives.squared_error(outputs, inputs))
        
    # prepare and compile training function
    params = lasagne.layers.get_all_params(net, trainable=True)
    initial_eta = 0.01
    eta_decay = 0.85
    momentum = 0.95
    eta = theano.shared(lasagne.utils.floatX(initial_eta))
    updates = lasagne.updates.nesterov_momentum(cost, params, eta, momentum)
    print("Compiling training function...")
    train_fn = theano.function([input_var_deconv, input_var], cost, updates=updates)
        
    
    # run training loop
    print("Training:")
    epochs = 20
    epochsize = 20
    batches = iter(batches)
    for epoch in range(epochs):
        err = 0
        for batch in progress(
                range(epochsize), min_delay=.5,
                desc='Epoch %d/%d: Batch ' % (epoch + 1, epochs)):
            data, labels = next(batches) # followed a simple styple *next(batches) from Jan is unclear what is passed.
            pred = pred_fn(data)    # a theano function returns a numpy array always. Here it is a column vector of shape 32  x 1
            err += train_fn(pred, data)
            
            '''# printing the weights of the first dense layer after each update            
            params = lasagne.layers.get_all_param_values(l_dense1)
            print(params[12])'''

            
            if not np.isfinite(err):
                print("\nEncountered NaN loss in training. Aborting.")
                sys.exit(1)
        print("Train loss: %.3f" % (err / epochsize))
        eta.set_value(eta.get_value() * lasagne.utils.floatX(eta_decay))

    # save final network
    print("Saving final model")
    np.savez(args.outfile, **{'param%d' % i: p for i, p in enumerate(
            lasagne.layers.get_all_param_values(net))})


if __name__ == '__main__':
    main()
