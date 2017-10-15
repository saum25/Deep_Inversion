#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Created on 6 Sep 2017

Code to generate a mel-spectrogram/spectrogram from an input
feature vector at layer l of the neural network

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
import matplotlib.pyplot as plt
import librosa.display as disp

def dist_euclidean(x, y):
    '''t1 = (x - y)
    t2 = t1 ** 2
    t3 = np.sum(t2)
    t4 = np.sqrt(t3)'''
    return np.sqrt(np.sum((x-y)**2))


def argument_parser():
    parser = argparse.ArgumentParser(description='Generates a mel-spectrogram from an input feature representation')
    parser.add_argument('modelfile', action='store', help='file to load the learned weights from (.npz format)')
    parser.add_argument('generatorfile', action='store', help='file to load the learned generator weights from (.npz format)')
    parser.add_argument('--dataset', default='jamendo', help='Name of the dataset to use.')
    parser.add_argument('--cache-spectra', metavar='DIR', default=None, help='Store spectra in the given directory (disabled by default).')
    '''parser.add_argument('--augment', action='store_true', default=True, help='If given, perform train-time data augmentation.')
    parser.add_argument('--no-augment', action='store_false', dest='augment', help='If given, disable train-time data augmentation.') '''

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
    with io.open(os.path.join(datadir, 'filelists', 'test')) as f:
        filelist = [l.rstrip() for l in f if l.rstrip()]

    # compute spectra
    print("Computing%s spectra..." %
          (" or loading" if args.cache_spectra else ""))
    spects = []
    for fn in progress(filelist, 'File '):
        cache_fn = (args.cache_spectra and os.path.join(args.cache_spectra, fn + '.npy'))
        spects.append(cached(cache_fn, audio.extract_spect, os.path.join(datadir, 'audio', fn),sample_rate, frame_len, fps))
    
    '''# load and convert corresponding labels
    print("Loading labels...")

    labels = []
    for fn, spect in zip(filelist, spects):
        fn = os.path.join(datadir, 'labels', fn.rsplit('.', 1)[0] + '.lab')
        with io.open(fn) as f:
            segments = [l.rstrip().split() for l in f if l.rstrip()]
        segments = [(float(start), float(end), label == 'sing')
                    for start, end, label in segments]
        timestamps = np.arange(len(spect)) / float(fps)
        labels.append(create_aligned_targets(segments, timestamps, np.bool))'''
        
    # prepare mel filterbank
    filterbank = audio.create_mel_filterbank(sample_rate, frame_len, mel_bands,
                                             mel_min, mel_max)
    filterbank = filterbank[:bin_mel_max].astype(floatX)    
    
    # precompute mel spectra, if needed, otherwise just define a generator
    mel_spects = (np.log(np.maximum(np.dot(spect[:, :bin_mel_max], filterbank),1e-7)) for spect in spects) # it is not clear why is Jan using natural log in place of log 10.
    
    '''if not args.augment:
        mel_spects = list(mel_spects)
        del spects'''
        
    # - load mean/std or compute it, if not computed yet
    meanstd_file = os.path.join(os.path.dirname(__file__), '%s_meanstd.npz' % args.dataset)
    with np.load(meanstd_file) as f:
            mean = f['mean']
            std = f['std']
    mean = mean.astype(floatX)
    istd = np.reciprocal(std).astype(floatX)
    
    # - prepare training data generator
    print("Preparing training data feed...")
    #if not args.augment:
        # Without augmentation, we just precompute the normalized mel spectra
        # and create a generator that returns mini-batches of random excerpts
    mel_spects = [(spect - mean) * istd for spect in mel_spects]

    '''else:
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
                    in_processes=True)'''
    
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
                network['fc9'], [f['param%d' % i] for i in range(len(f.files))])

    # create output expression
    outputs_score = lasagne.layers.get_output(network['conv4'], deterministic=True)

    # prepare and compile prediction function
    print("Compiling prediction function...")
    pred_fn = theano.function([input_var], outputs_score)
    
    # training the Upconvolutional network - Network 2
    
    #input_var_deconv = T.matrix('input_var_deconv')
    input_var_deconv = T.tensor4('input_var_deconv')
    #inputs_deconv = input_var_deconv.dimshuffle(0, 1, 'x', 'x') # 32x 1 x 1 x 1. Adding the width and depth dimensions
    #gen_network = upconv.architecture_upconv_fc8(input_var_deconv, (batchsize, lasagne.layers.get_output_shape(network['fc8'])[1]))
    gen_network = upconv.architecture_upconv_conv4(input_var_deconv, (batchsize, lasagne.layers.get_output_shape(network['conv4'])[1], lasagne.layers.get_output_shape(network['conv4'])[2], lasagne.layers.get_output_shape(network['conv4'])[3]))
    
    # load saved weights
    with np.load(args.generatorfile) as f:
        lasagne.layers.set_all_param_values(
                gen_network, [f['param%d' % i] for i in range(len(f.files))])

    
    # create cost expression
    outputs = lasagne.layers.get_output(gen_network, deterministic=True)
    print("Compiling training function...")
    test_fn = theano.function([input_var_deconv], outputs)
        
    
    # run prediction loop
    print("Predicting:")
    # we select n_excerpts per input audio file sequentially after randomly shuffling the indices of excerpts
    n_excerpts = 32
    # array of n_excerpts randomly chosen excerpts
    sampled_excerpts = np.zeros((len(filelist) * n_excerpts, blocklen, spect.shape[1]))
    # list of n_excerpts (randomly chosen) per file 
    #input_melspects = []
    counter = 0
    
    # we calculate the reconstruction error for five random draws and the final value is average of it.
    iterations = 10
    n_count = 0
    avg_error_n = 0

    while (iterations):
        counter = 0
        print("========")
        print("Interation number:%d"%(n_count+1))
        for spect in progress(mel_spects, total=len(filelist), desc='File '):
            # Step 1: From each mel spectrogram we create excerpts of length 115 frames
            num_excerpts = len(spect) - blocklen + 1
            # excerpts is a numpy array of shape num_excerpts x blocklen x spect.shape[1]
            excerpts = np.lib.stride_tricks.as_strided(spect, shape=(num_excerpts, blocklen, spect.shape[1]), strides=(spect.strides[0], spect.strides[0], spect.strides[1]))
    
            # Step 2: Select n_excerpts randomly from all the excerpts
            indices = range(len(excerpts))
            np.random.shuffle(indices)
            for i in range(n_excerpts):
                sampled_excerpts[i + counter*n_excerpts] = excerpts[indices[i]]
            #input_melspects.append(sampled_excerpts)
            counter +=1
            
        # evaluating the normalising constant (N), which is given by the average pairwise euclidean distance between the randomly chosen samples in the test/validation set
        dist_matrix = np.zeros((len(filelist) * n_excerpts, len(filelist) * n_excerpts))
        
        for i in range(len(sampled_excerpts)):
            for j in range(len(sampled_excerpts)):
                dist_matrix [i][j]= dist_euclidean(sampled_excerpts[i], sampled_excerpts[j])
                
        # print(dist_matrix)
        # dist_matrix is a symmetric matrix
        # Denominator to calculate the average needs to be considering 'n' zero values in a n x n symmetric matrix
        # Such numbers don't add anything to the sum, hence to be removed from the denominator.
        # n*n - n is the number in the denominator
        d = len(sampled_excerpts)
        N = (np.sum(dist_matrix))/(d * (d-1))
        print("Normalization constant: %f" %(N))
        
        # generating spectrums from feature representations
        
        # Step 1: first generate a feature representation from input, i.e. randomly selected spectrogram
        # each iteration returns a matrix of shape: 32 x 64
        # preds shape: len(filelist) x n_excerpts, where each row correspond to features per spectrogram randomly sampled
        preds = []
        for pos in range(0, len(filelist) * n_excerpts, batchsize):
            preds.append((pred_fn(sampled_excerpts[pos:pos + batchsize])))
        #preds = np.vstack(preds)
        #print(preds)
    
        # Step 2 : passing all features per file to generate spectrogram
        # Theano function returns a 4d array, mini_batch_size x 1 x blocklen x mel_dimensions
        # mel_predictions_array shape: number of samples x blocklen x mel_dimensions , where number of samples = n_files * n excerpts per file
        mel_predictions = []
        for pos in range(len(preds)):
            mel_predictions.append(np.squeeze((test_fn(preds[pos])), axis = 1))
        #print(mel_predictions[0].shape)
        mel_predictions_array = np.concatenate(mel_predictions, axis=0)
        #print(mel_predictions_array.shape)
        
        # Step 3: Reconstruction error calculation per sample
        # we calculate the average normalised reconstruction error
        # an assumption that order of samples is maintained.
        error_n = 0
        for i in range(len(sampled_excerpts)):
            error = dist_euclidean(sampled_excerpts [i], mel_predictions_array[i])
            error_n += error
        error_n = error_n/N
        avg_error_n = error_n/len(sampled_excerpts) + avg_error_n
        print("average normalised reconstruction error:%f iteration: %d" %(error_n/len(sampled_excerpts), n_count+1))
        
        iterations -=1
        n_count+=1
    print('======')    
    print("Average normalised reconstruction error:%f after %d iteration" %(avg_error_n/n_count, n_count))
    
    # plotting a fixed selected input excerpt and its reconstruction after inversion from the uponv network
    print("Plotting the excerpt's reconstruction")
    num_excerpts = len(mel_spects[0]) - blocklen + 1
    # excerpts is a numpy array of shape num_excerpts x blocklen x spect.shape[1]
    excerpts = np.lib.stride_tricks.as_strided(mel_spects[0], shape=(num_excerpts, blocklen, mel_spects[0].shape[1]), strides=(mel_spects[0].strides[0], mel_spects[0].strides[0], mel_spects[0].strides[1]))
    
    preds = []
    pos = 100
    preds.append((pred_fn(excerpts[pos:pos + batchsize])))
    
    mel_predictions = []
    mel_predictions.append(np.squeeze((test_fn(preds[0])), axis = 1))
    #print(mel_predictions[0].shape)

    excerpt_index = pos
    '''fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    ax1.imshow(sampled_excerpts[excerpt_index].T, vmin=-3, cmap='jet', aspect='auto',
               interpolation='nearest')
    ax1.set_title('Input Mel-spectrogram')
    print(sampled_excerpts[excerpt_index])
    print(mel_predictions_array[excerpt_index])
    ax2.imshow(mel_predictions_array[excerpt_index].T, vmin=-3, cmap='jet', aspect='auto',
               interpolation='nearest')
    ax2.set_title('Re-generated Mel-spectrogram')
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    fig.suptitle('Plots for excerpt index: %d' %(excerpt_index), fontsize=16)'''
    
    print("Error in generating the displayed instance: %f" %(dist_euclidean(excerpts[excerpt_index], mel_predictions[0][0])))
    
    plt.subplot(2, 1, 1)
    disp.specshow(excerpts[excerpt_index].T, y_axis='mel', hop_length= 315, x_axis='time', fmin=27.5, fmax=8000)
    plt.colorbar()
    plt.subplot(2, 1, 2)
    disp.specshow(mel_predictions[0][0].T, y_axis='mel', hop_length= 315, x_axis='time', fmin=27.5, fmax=8000)
    plt.colorbar()
    plt.suptitle('Plots for excerpt index: %d' %(excerpt_index), fontsize=16)
    #print(sampled_excerpts[excerpt_index])
    #print(mel_predictions_array[excerpt_index])   
    
    plt.show()



if __name__ == '__main__':
    main()
