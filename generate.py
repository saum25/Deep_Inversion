#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Created on 6 Sep 2017

Code to generate a mel-spectrogram/spectrogram from an input
feature vector at layer l of the neural network

@author: Saumitra
'''

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
import numpy.linalg as linalg
import util
import plots
import randomise
import analysis

def main():
    
    parser = util.argument_parser()
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
    
    spects = [] # list of tuple, where each tuple has magnitude and phase information for one audio file
    for fn in progress(filelist, 'File '):
        cache_fn = (args.cache_spectra and os.path.join(args.cache_spectra, fn + '.npy'))
        spects.append(cached(cache_fn, audio.extract_spect, os.path.join(datadir, 'audio', fn),sample_rate, frame_len, fps))
    
    spects_mag = [ spect[0] for spect in spects]    # magnitude per audio file
    spects_phase = [ spect[1] for spect in spects]  # phase per audio file
        
    # prepare mel filterbank
    filterbank = audio.create_mel_filterbank(sample_rate, frame_len, mel_bands,
                                             mel_min, mel_max)
    filterbank = filterbank[:bin_mel_max].astype(floatX)
    
    # pseudo-inverse: used for inverting Mel=spectrogram back to audio.
    filterbank_pinv = linalg.pinv(filterbank)
    
    # precompute mel spectra, if needed, otherwise just define a generator
    mel_spects = (np.log(np.maximum(np.dot(spect[:, :bin_mel_max], filterbank),1e-7)) for spect in spects_mag) # it is not clear why is Jan using natural log in place of log 10.
            
    # load mean/std or compute it, if not computed yet
    meanstd_file = os.path.join(os.path.dirname(__file__), '%s_meanstd.npz' % args.dataset)
    with np.load(meanstd_file) as f:
            mean = f['mean']
            std = f['std']
    mean = mean.astype(floatX)
    istd = np.reciprocal(std).astype(floatX)
    
    # prepare training data generator
    print("Preparing training data feed...")
    # Without augmentation, we just precompute the normalized mel spectra
    # and create a generator that returns mini-batches of random excerpts
    mel_spects = [(spect - mean) * istd for spect in mel_spects]
    #print(mel_spects[0].shape)

    print("Preparing training functions...")
    # we create two functions by using two network architectures. One uses the pre-trained discriminator network
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
    outputs_score = lasagne.layers.get_output(network['fc8'], deterministic=True)
    outputs_pred = lasagne.layers.get_output(network['fc9'], deterministic=True)

    # prepare and compile prediction function
    print("Compiling prediction function...")
    pred_fn_score = theano.function([input_var], outputs_score, allow_input_downcast= True)
    pred_fn = theano.function([input_var], outputs_pred, allow_input_downcast= True)
    
    # training the Upconvolutional network - Network 2    
    input_var_deconv = T.matrix('input_var_deconv')
    #input_var_deconv = T.tensor4('input_var_deconv')
    gen_network = upconv.architecture_upconv_fc8(input_var_deconv, (batchsize, lasagne.layers.get_output_shape(network['fc8'])[1]))
    #gen_network = upconv.architecture_upconv_c1(input_var_deconv, (batchsize, lasagne.layers.get_output_shape(network['conv1'])[1], lasagne.layers.get_output_shape(network['conv1'])[2], lasagne.layers.get_output_shape(network['conv1'])[3]), args.n_conv_layers, args.n_conv_filters)
    
    # load saved weights
    with np.load(args.generatorfile) as f:
        lasagne.layers.set_all_param_values(
                gen_network, [f['param%d' % i] for i in range(len(f.files))])
    
    # create cost expression
    outputs = lasagne.layers.get_output(gen_network, deterministic=True)
    print("Compiling generator function...")
    test_fn = theano.function([input_var_deconv], outputs, allow_input_downcast= True)        
    
    # run prediction loop
    print("Predicting:")
    # we select n_excerpts per input audio file sequentially after randomly shuffling the indices of excerpts
    n_excerpts = 32
    # array of n_excerpts randomly chosen excerpts
    sampled_excerpts = np.zeros((len(filelist) * n_excerpts, blocklen, mel_bands))
      
    # we calculate the reconstruction error for 'iterations' random draws and the final value is average of it.
    iterations = 1
    n_count = 0
    avg_error_n = 0
    counter = 0
    print("Number of elements (extracted from each file): %d" %(len(mel_spects)))

    while (iterations):
        counter = 0
        print("========")
        print("Interation number:%d"%(n_count+1))
        for spect in mel_spects:
            # Step 1: From each mel spectrogram we create excerpts of length blocklen frames
            num_excerpts = len(spect) - blocklen + 1
            # excerpts is a numpy array of shape num_excerpts x blocklen x spect.shape[1]: not sure what spect.strides capture??
            excerpts = np.lib.stride_tricks.as_strided(spect, shape=(num_excerpts, blocklen, spect.shape[1]), strides=(spect.strides[0], spect.strides[0], spect.strides[1]))
    
            # Step 2: Select n_excerpts randomly from all the excerpts
            indices = range(len(excerpts))
            np.random.shuffle(indices)
            for i in range(n_excerpts):
                sampled_excerpts[i + counter*n_excerpts] = excerpts[indices[i]]
            #input_melspects.append(sampled_excerpts)
            counter +=1
            
        print('Shape of the randomly sampled 3-d excerpt array')
        print(sampled_excerpts.shape)    
        
        # evaluating the normalising constant (N), which is given by the average pairwise euclidean distance between the randomly chosen samples in the test set
        dist_matrix = np.zeros((len(filelist) * n_excerpts, len(filelist) * n_excerpts))
        
        for i in range(len(sampled_excerpts)):
            for j in range(len(sampled_excerpts)):
                dist_matrix [i][j]= util.dist_euclidean(sampled_excerpts[i], sampled_excerpts[j])
                
        # print(dist_matrix)
        # dist_matrix is a symmetric matrix
        # Denominator to calculate the average needs to be considering 'n' zero values in a n x n symmetric matrix
        # Such numbers don't add anything to the sum, hence to be removed from the denominator.
        # n*n - n is the number in the denominator
        d = len(sampled_excerpts)
        N = (np.sum(dist_matrix))/(d * (d-1))
        print("Normalization constant: %f" %(N))
        
        # generating spectrums from feature representations
        
        # Step 1: first generate a feature representation for the input, i.e. randomly selected spectrogram using the pre-trained network
        # each iteration returns a matrix of shape: 32 x 64
        # in preds each element is a matrix of shape batch_size x 64, where each row correspond to features per spectrogram randomly sampled
        preds = []
        for pos in range(0, len(filelist) * n_excerpts, batchsize):
            preds.append((pred_fn_score(sampled_excerpts[pos:pos + batchsize])))
    
        # Step 2 : passing all features per file to generate spectrogram
        # Theano function returns a 4d array, mini_batch_size x 1 x blocklen x mel_dimensions
        # mel_predictions_array shape: number of samples x blocklen x mel_dimensions , where number of samples = n_files * n excerpts per file
        mel_predictions = []
        for pos in range(len(preds)):
            mel_predictions.append(np.squeeze((test_fn(preds[pos])), axis = 1))
        mel_predictions_array = np.concatenate(mel_predictions, axis=0)
        #print(mel_predictions_array.shape)
        
        # Step 3: Reconstruction error calculation per sample
        # we calculate the average normalised reconstruction error
        # an assumption that order of samples is maintained.
        error_n = 0
        for i in range(len(sampled_excerpts)):
            error = util.dist_euclidean(sampled_excerpts [i], mel_predictions_array[i])
            error_n += error
        error_n = error_n/N
        avg_error_n = error_n/len(sampled_excerpts) + avg_error_n
        print("average normalised reconstruction error:%f iteration: %d" %(error_n/len(sampled_excerpts), n_count+1))
        
        iterations -=1
        n_count+=1
    print('==========')    
    print("Average normalised reconstruction error:%f after %d iteration" %(avg_error_n/n_count, n_count))
    
    #------------------------------------------------------------------------# 
    # code for instance-based feature inversion and analysis
    # (1) Pick a file from dataset (dataset: Jamendo test) (2) Select a time index to read from
    
    print('\n===Instance based analysis====\n')
    start_offset = 10
    end_offset = 20 
    duration =  180  # secs
    mask_threshold = [0.5, 0.6, 0.7]#np.linspace(0, 1.1, 12)    # gives 1 as an extra value
    class_threshold = 0.66  # Jan's code
    error_threshold = 0.5
    file_idx = np.arange(0, len(filelist))    
    hop_size= sample_rate/fps # samples
    dump_path = './audio'   # path to save the reconstructed audio
    pred_before = []    # class predictions before masking
    pred_after = [] # class predictions after masking
    gen_error = []  # error in generation per instance
    area_per_instance = [] # area covered in the generated mask based on recon
    plot_flag = False
    result = []     # finally a list of tuples is created in the order
                    # threshold, total_instances, total_fail, average_area, cc_lt, cnc_lt, cc_gt, cnc_gt
    ms_z_norm = True # standard score based normalisation of each bin after masking
    debug_flag = True
    abs_pred_error = []
    
    for mt in mask_threshold:
        print("\n ++++++Analysis for the mask threshold: %f +++++\n " %(mt))
        for file_instance in file_idx:
            print("<<<<Analysis for the file: %d>>>>" %(file_instance+1))
            time_idx = np.random.randint(start_offset, end_offset, 1)[0]   # secs # tells given the offset, what frame_idx should it match?
            td = time_idx
            
            # re-generating all the excerpts for the selected file_idx
            # excerpts is a 3-d array of shape: num_excerpts x blocklen x mel_spects_dimensions   
            num_excerpts = len(mel_spects[file_instance]) - blocklen + 1
            print("Number of excerpts in the file :%d" %num_excerpts)
            excerpts = np.lib.stride_tricks.as_strided(mel_spects[file_instance], shape=(num_excerpts, blocklen, mel_spects[file_instance].shape[1]), strides=(mel_spects[file_instance].strides[0], mel_spects[file_instance].strides[0], mel_spects[file_instance].strides[1]))
            
            while(time_idx<= td+duration):
                # convert the time_idx to the excerpt index for the reconstruction
                excerpt_idx = int(np.round((time_idx * sample_rate)/(hop_size)))
                print("Time_idx: %f secs, Excerpt_idx: %d" %(time_idx, excerpt_idx))
    
                if ((excerpt_idx +  blocklen) > num_excerpts):
                    print("------------------Number of excerpts are less for file: %d--------------------" %(file_instance+1))
                    break                      
                
                # reconstructing the selected spectrogram segment, that starts at the time_idx and is of length blocklen
                # done to make sure the reconstruction works fine, and the time and frame indices are mapped correctly.
                sub_matrix_mag = spects_mag[file_instance][excerpt_idx:excerpt_idx+blocklen]
                sub_matrix_phase = spects_phase[file_instance][excerpt_idx:excerpt_idx+blocklen]
                util.recon_audio(sub_matrix_mag, sub_matrix_phase, dump_path,'spect_', frame_len, sample_rate/fps, sample_rate)  
                
                # generating feature representations for the chosen excerpt.
                # CAUTION: Need to feed mini-batch to pre-trained model, so (mini_batch-1) following excerpts are also fed, but are not analysed
                scores = pred_fn_score(excerpts[excerpt_idx:excerpt_idx + batchsize])
                #print("Feature representation")
                #print(scores[file_idx])
                predictions = pred_fn(excerpts[excerpt_idx:excerpt_idx + batchsize])
                print("Predictions score for the excerpt before masking:%f" %(predictions[0]))
                pred_before.append(predictions[0][0])
                                   
                mel_predictions = np.squeeze(test_fn(scores), axis = 1) # mel_predictions is a 3-d array of shape batch_size x blocklen x n_mels
                error_instance = util.dist_euclidean(excerpts[excerpt_idx], mel_predictions[0])/N
                gen_error.append(error_instance)
                print("NRE in generating the instance: %f" %((error_instance)))
                    
                # normalising the inverted mel to create a map, and use the map to cut the section in the input mel
                norm_inv = util.normalise(mel_predictions[0])
                norm_inv[norm_inv<mt] = 0 # Binary mask----- 
                norm_inv[norm_inv>=mt] = 1
                    
                # randomisation or not
                norm_inv, area = randomise.random_selection(norm_inv, random_block = False, debug_print = False)
                
                # reversing the mask to keep the portions that seem not useful for the current instance prediction
                '''for i in range(norm_inv.shape[0]):
                    for j in range(norm_inv.shape[1]):
                        if norm_inv[i][j]==0:
                            norm_inv[i][j]=1
                        else:
                            norm_inv[i][j]=0'''
        
                # masking out the input based on the mask created above
                masked_input = np.zeros((batchsize, blocklen, mel_bands))
                if ms_z_norm:
                    unnorm_excerpt = (excerpts[excerpt_idx]/istd) + mean    # removing mean scaling as we want to renormalise latter
                    masked_input[0] = norm_inv * unnorm_excerpt # only fill the instance thats being analysed
                    masked_input = (masked_input - mean)*istd
                else:
                    masked_input[0] = norm_inv * excerpts[excerpt_idx]
                    
                plots.plot_figures(util.normalise(excerpts[excerpt_idx]), util.normalise(mel_predictions[0]), norm_inv, util.normalise(masked_input[0]), excerpt_idx, plot_flag)
                
                # reconstructing the input mel-spectrogram excerpt
                masked_spect = util.preprocess_recon(excerpts[excerpt_idx], filterbank_pinv, spects_mag [file_instance] [excerpt_idx:excerpt_idx+blocklen, bin_mel_max:bin_nyquist], istd, mean)
                util.recon_audio(masked_spect, spects_phase[file_instance][excerpt_idx:blocklen+excerpt_idx], dump_path,'mel_',frame_len, sample_rate/fps, sample_rate)
                
                # reconstructing masked out version of input mel spectrogram
                masked_spect = util.preprocess_recon(masked_input[0], filterbank_pinv, spects_mag [file_instance] [excerpt_idx:excerpt_idx+blocklen, bin_mel_max:bin_nyquist], istd, mean)
                util.recon_audio(masked_spect, spects_phase[file_instance][excerpt_idx:blocklen+excerpt_idx], dump_path,'mask_inv_',frame_len, sample_rate/fps, sample_rate)
                
                time_idx += 1   # shifting the time window by 1 sec
                
                # create input data after masking the previous input and feed it to the network.
                # just changing the first input.
                predictions = pred_fn(masked_input)
                print("Predictions score for the excerpt after masking:%f\n" %(predictions[0]))
                pred_after.append(predictions[0][0])
                
                # area per instance
                area_per_instance.append(area)

        abs_error, res_tuple = analysis.result_analysis(pred_before, pred_after, area_per_instance, gen_error, mt, class_threshold, error_threshold, debug_flag)        

        result.append(res_tuple)
        abs_pred_error.append(abs_error)
        
        # clearing the lists : couldn't find a better way
        pred_before = []
        pred_after = []
        gen_error = []
        area_per_instance = []

    # saving the absolute change in prediction error
    np.savez('pred_err.npz', **{'mt_%d'%i:abs_pred_error[i] for i in range(len(abs_pred_error))})
    
    # save the final results
    with open('result.txt', 'w') as fp:
        fp.write('\n'.join('{} {} {} {} {} {} {} {}'.format(x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7]) for x in result))

if __name__ == '__main__':
    main()
