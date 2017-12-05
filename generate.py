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
import numpy.linalg as linalg
import librosa

def dist_euclidean(x, y):
    '''t1 = (x - y)
    t2 = t1 ** 2
    t3 = np.sum(t2)
    t4 = np.sqrt(t3)'''
    return np.sqrt(np.sum((x-y)**2))

def dist_euclidean_matrix(x, y):
    return ((x-y)**2)

def normalise(x):
    """
    Normalise a vector/ matrix, in range 0 - 1
    """
    return((x-x.min())/(x.max()-x.min()))


def argument_parser():
    parser = argparse.ArgumentParser(description='Generates a mel-spectrogram from an input feature representation')
    parser.add_argument('modelfile', action='store', help='file to load the learned weights from (.npz format)')
    parser.add_argument('generatorfile', action='store', help='file to load the learned generator weights from (.npz format)')
    parser.add_argument('--dataset', default='jamendo', help='Name of the dataset to use.')
    parser.add_argument('--cache-spectra', metavar='DIR', default=None, help='Store spectra in the given directory (disabled by default).')
    '''parser.add_argument('--augment', action='store_true', default=True, help='If given, perform train-time data augmentation.')
    parser.add_argument('--no-augment', action='store_false', dest='augment', help='If given, disable train-time data augmentation.') '''
    parser.add_argument('--n_conv_layers', default=1, type=int, help='number of 3x3 conv layers to be added before upconv layers')
    parser.add_argument('--n_conv_filters', default=32, type=int, help='number of filters per conv layer in the upconvolutonal architecture for Conv layer inversion')

    return parser

def preprocess_recon(masked_mel, filtbank_pinv, spect_mag, istd, mean):
    """
    converts the Mel-spectrogram magnitude matrix to spectrogram matrix
    """
    masked_spect = np.dot(np.exp((masked_mel/istd)+mean), filtbank_pinv) # blocklen * bin_mel_max(372 in this case)
    masked_spect = np.concatenate((masked_spect, spect_mag), axis =1)
    return masked_spect    

def recon_audio(spect_mag, spect_phase, pathname, filetag, n_fft, hop_len, sampling_rate):
    """
    takes in magnitude and phase components, and recreates the temporal signal
    """
    # complex valued array
    spect_recon = spect_mag * spect_phase
    
    # inverting    
    win_len = n_fft
    ifft_window = np.hanning(win_len)
    
    n_frames = spect_recon.T.shape[1]
    expected_signal_len = n_fft + hop_len * (n_frames - 1)   # How? but important
    audio_recon = np.zeros(expected_signal_len)
        
    for i in range(n_frames):
        sample = i * hop_len
        spec = spect_recon.T[:, i].flatten()
        spec = np.concatenate((spec.conj(), spec[-2:0:-1]), 0)  # not clear? but expands the 513 input to 1024 as DFT is symmetric
        ytmp = ifft_window * np.fft.irfft(spec, n = n_fft)

        audio_recon[sample:(sample + n_fft)] = audio_recon[sample:(sample + n_fft)] + ytmp
    
    # not sure why the librosa recon is not working. Should work??
    #audio_recon = librosa.core.istft( spect_recon.T, hop_length = 315, win_length = 1024, center = False, window = np.hanning)
    librosa.output.write_wav(os.path.join(pathname, filetag +'recon_excerpt.wav'), audio_recon, sampling_rate, norm = True)



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
    
    spects = [] # list of tuple, where each tuple has magnitude and phase information for one audio file
    for fn in progress(filelist, 'File '):
        cache_fn = (args.cache_spectra and os.path.join(args.cache_spectra, fn + '.npy'))
        spects.append(cached(cache_fn, audio.extract_spect, os.path.join(datadir, 'audio', fn),sample_rate, frame_len, fps))
    
    spects_mag = [ spect[0] for spect in spects]    # magnitude per audio file
    spects_phase = [ spect[1] for spect in spects]  # phase per audio file
    print(spects_mag[0][0])
    
        
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
    print(mel_spects[0].shape)

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
    pred_fn_score = theano.function([input_var], outputs_score, allow_input_downcast = True)
    pred_fn = theano.function([input_var], outputs_pred, allow_input_downcast = True)
    
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
    print("Compiling training function...")
    test_fn = theano.function([input_var_deconv], outputs, allow_input_downcast=True)        
    
    # run prediction loop
    print("Predicting:")
    # we select n_excerpts per input audio file sequentially after randomly shuffling the indices of excerpts
    n_excerpts = 32
    # array of n_excerpts randomly chosen excerpts
    sampled_excerpts = np.zeros((len(filelist) * n_excerpts, blocklen, mel_bands))
      
    # we calculate the reconstruction error for 'iterations' random draws and the final value is average of it.
    iterations = 4
    n_count = 0
    avg_error_n = 0
    counter = 0
    print(mel_spects[0][0])

    while (iterations):
        counter = 0
        print("========")
        print("Interation number:%d"%(n_count+1))
        for spect in progress(mel_spects, total=len(filelist), desc='File '):
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
            
        #print('Shape of the randomly sampled 3-d excerpt array')
        #print(sampled_excerpts.shape)    
        
        # evaluating the normalising constant (N), which is given by the average pairwise euclidean distance between the randomly chosen samples in the test set
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
	print(np.sum(dist_matrix))
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
            error = dist_euclidean(sampled_excerpts [i], mel_predictions_array[i])
            error_n += error
        error_n = error_n/N
        avg_error_n = error_n/len(sampled_excerpts) + avg_error_n
        print("average normalised reconstruction error:%f iteration: %d" %(error_n/len(sampled_excerpts), n_count+1))
        
        iterations -=1
        n_count+=1
    print('======')    
    print("Average normalised reconstruction error:%f after %d iteration" %(avg_error_n/n_count, n_count))
    
    #------------------------------------------------------------------------# 
    # code for instance-based feature inversion and analysis
    # (1) Pick a file from dataset (dataset: Jamendo test) (2) Select an time index to read from
    
    print('\r ===Instance based analysis==== \r')
    
    file_idx = np.arange(0, 16)
    time_idx = 20# secs # tells given the offset, what frame_idx should it match? 
    hop_size= sample_rate/fps # samples
    dump_path = './audio'   # path to save reconstructed audio
    pred_before = []
    pred_after = []
    for file_instance in file_idx:
        print("\r Analysis for file idx: %d" %file_instance)
        time_idx = 20
        while(time_idx<24):
            # convert time_idx to excerpt index for reconstruction
            excerpt_idx = int(np.round((time_idx * sample_rate)/(hop_size)))
            print("\r Excerpt_idx: %d, Time_idx: %f secs" %(excerpt_idx, time_idx))
            
            # reconstructing the selected spectrogram segment, that starts at time_idx and is of length blocklen
            # done to make sure the reconstruction works fine, and the time and frame indices are mapped correctly.
            sub_matrix_mag = spects_mag[file_instance][excerpt_idx:excerpt_idx+blocklen]
            sub_matrix_phase = spects_phase[file_instance][excerpt_idx:excerpt_idx+blocklen]
            recon_audio(sub_matrix_mag, sub_matrix_phase, dump_path,'spect_', frame_len, sample_rate/fps, sample_rate)  
            
            # re-generating all the excerpts for the selected file_idx
            # excerpts is a 3-d array of shape: num_excerpts x blocklen x mel_spects_dimensions   
            # print("Plotting the excerpt's reconstruction")
            num_excerpts = len(mel_spects[file_instance]) - blocklen + 1
            print("Number of excerpts in the file :%d" %num_excerpts)
            excerpts = np.lib.stride_tricks.as_strided(mel_spects[file_instance], shape=(num_excerpts, blocklen, mel_spects[file_instance].shape[1]), strides=(mel_spects[file_instance].strides[0], mel_spects[file_instance].strides[0], mel_spects[file_instance].strides[1]))
            
            # generating feature representations for the chosen excerpt.
            # CAUTION: Need to feed mini-batch to pre-trained model, so (mini_batch-1) following excerpts are also fed.
            scores = pred_fn_score(excerpts[excerpt_idx:excerpt_idx + batchsize])
            #print("Feature representation")
            #print(scores[file_idx])
            predictions = pred_fn(excerpts[excerpt_idx:excerpt_idx + batchsize])
            print("Predictions score for the excerpt is:%f" %(predictions[0]))
            pred_before.append(predictions[0][0])
            
            # binarisation
            '''n_nz = np.count_nonzero(scores[0])
            norm_pred = linalg.norm(scores[0])
            replace = norm_pred/(np.sqrt(n_nz))
            print("Non zero values: %d, L2-Norm:%f replacement: %f" %(n_nz, norm_pred, replace))
            binarised_scores = np.empty(len(scores[0]))
            for i in range(len(scores[0])):    # write efficient code TBC
                if scores[0][i]>0:
                    binarised_scores[i] = replace
                elif scores[0][i]<0:
                    binarised_scores[i] = -replace
                else:
                    pass
            print("Binarized feature representation")
            scores[0]=binarised_scores
            print(scores[0])'''
                    
            mel_predictions = np.squeeze(test_fn(scores), axis = 1) # mel_predictions is a 3-d array of shape batch_size x blocklen x n_mels
            
            print("Error in generating the instance: %f" %(dist_euclidean(excerpts[excerpt_idx], mel_predictions[0])))
                
            # Input (Unnormalised Mel spectrogram)
            '''plt.figure(1)
            plt.subplot(2, 3, 1)
            disp.specshow(excerpts[excerpt_idx].T, y_axis='mel', hop_length= 315, x_axis='time', fmin=27.5, fmax=8000, cmap = 'coolwarm')
            plt.title('Input Mel-spectrogram')
            plt.xlabel('Time')
            plt.ylabel('Hz')
            plt.colorbar()
            
            # Input, normalised and thresholded mel-spectrogram (top25%)
            plt.subplot(2, 3, 2)
            disp.specshow(((normalise(excerpts[excerpt_idx]))>0.75).T, y_axis='mel', hop_length= 315, x_axis='time', fmin=27.5, fmax=8000, cmap = 'gray_r')
            plt.title('Input Mel-spectrogram (N_T_top25%)')
            plt.xlabel('Time')
            plt.ylabel('Hz')
            plt.colorbar()
            
            # Input, normalised and thresholded mel-spectrogram (top35%)
            plt.subplot(2, 3, 3)
            disp.specshow(((normalise(excerpts[excerpt_idx]))>0.65).T, y_axis='mel', hop_length= 315, x_axis='time', fmin=27.5, fmax=8000, cmap = 'gray_r')
            plt.title('Input Mel-spectrogram (N_T_top35%)')
            plt.xlabel('Time')
            plt.ylabel('Hz')
            plt.colorbar()
            
            # Inverted feature representation
            plt.subplot(2, 3, 4)
            plt.title('Inversion from FC8')
            disp.specshow(mel_predictions[0].T, y_axis='mel', hop_length= 315, x_axis='time', fmin=27.5, fmax=8000, cmap='coolwarm')
            plt.xlabel('Time')
            plt.ylabel('Hz')
            plt.colorbar()
            
            # Normalised and thresholded inverted feature representation (top 25%)
            plt.subplot(2, 3, 5)
            plt.title('Inversion from FC8((N_T_top25%))')
            disp.specshow(((normalise(mel_predictions[0]))>0.75).T, y_axis='mel', hop_length= 315, x_axis='time', fmin=27.5, fmax=8000, cmap='gray_r')
            plt.xlabel('Time')
            plt.ylabel('Hz')
            plt.colorbar()
    
            # Input, normalised and thresholded mel-spectrogram (top 35%)        
            plt.subplot(2, 3, 6)
            plt.title('Inversion from FC8((N_T_top35%))')
            disp.specshow(((normalise(mel_predictions[0]))>0.65).T, y_axis='mel', hop_length= 315, x_axis='time', fmin=27.5, fmax=8000, cmap='gray_r')
            plt.xlabel('Time')
            plt.ylabel('Hz')
            plt.colorbar()
            plt.suptitle('Plots for excerpt index: %d' %(excerpt_idx))'''
          
            # normalising the inverted mel, to create a map, and use the map to cut the section in the input mel
            # thresholded at 75%
            norm_inv = normalise(mel_predictions[0])
            norm_inv[norm_inv<0.50] = 0 # Binary mask----- 
            norm_inv[norm_inv>=0.50] = 1
            # reversing the mask to keep the portions that seem not useful for the current instance prediction
            '''for i in range(norm_inv.shape[0]):
                for j in range(norm_inv.shape[1]):
                    if norm_inv[i][j]==1e-7:
                        norm_inv[i][j]=1
                    else:
                        norm_inv[i][j]=1e-7'''
    
            '''plt.figure(2)
            plt.subplot(3, 1, 1)
            disp.specshow(excerpts[excerpt_idx].T, y_axis='mel', hop_length= 315, x_axis='time', fmin=27.5, fmax=8000, cmap = 'coolwarm')
            plt.title('Input Mel-spectrogram')
            plt.xlabel('Time')
            plt.ylabel('Hz')
            plt.colorbar()
            
            plt.subplot(3, 1, 2)
            disp.specshow(norm_inv.T, y_axis='mel', hop_length= 315, x_axis='time', fmin=27.5, fmax=8000, cmap = 'gray_r')
            plt.title('Mask based on normalised and thresholded inverted mel')
            plt.colorbar()
            #figure_name = args.generatorfile.rsplit('/', 2)
            #plt.savefig(figure_name[0]+'/'+figure_name[1]+'/'+figure_name[1]+'_ii100.pdf', dpi = 300)'''
    
            # masking out the input based on the mask created above
            masked_input = np.zeros((batchsize, blocklen, mel_bands))
            unnorm_excerpt = (excerpts[excerpt_idx]/istd) + mean    # removing mean scaling as we want to renormalise latter
            masked_input[0] = norm_inv * unnorm_excerpt # only fill the instance thats being analysed
            masked_input = (masked_input - mean)*istd
    
            '''plt.subplot(3, 1, 3)
            disp.specshow((masked_input[0]).T, y_axis='mel', hop_length= 315, x_axis='time', fmin=27.5, fmax=8000, cmap = 'coolwarm')
            plt.colorbar()
            plt.title('Masked input Mel spectrogram')
            plt.suptitle('Plots for excerpt index: %d' %(excerpt_idx))
            plt.show()'''
            
            # reconstructing the input mel-spectrogram excerpt
            masked_spect = preprocess_recon(excerpts[excerpt_idx], filterbank_pinv, spects_mag [file_instance] [excerpt_idx:excerpt_idx+blocklen, bin_mel_max:bin_nyquist], istd, mean)
            recon_audio(masked_spect, spects_phase[file_instance][excerpt_idx:blocklen+excerpt_idx], dump_path,'mel_',frame_len, sample_rate/fps, sample_rate)
            
            # reconstructing masked out version of input mel spectrogram
            masked_spect = preprocess_recon(masked_input[0], filterbank_pinv, spects_mag [file_instance] [excerpt_idx:excerpt_idx+blocklen, bin_mel_max:bin_nyquist], istd, mean)
            recon_audio(masked_spect, spects_phase[file_instance][excerpt_idx:blocklen+excerpt_idx], dump_path,'mask_inv_',frame_len, sample_rate/fps, sample_rate)
            
            time_idx +=0.5
            
            # create input data after masking the previous input and feed it to the network.
            # just changing the first input.
            predictions = pred_fn(masked_input)
            print("Predictions score for the excerpt is:%f %f %f" %(predictions[0], predictions[1], predictions[2]))
            pred_after.append(predictions[0][0])
            '''scores = pred_fn_score(masked_input)
            mel_predictions = np.squeeze(test_fn(scores), axis = 1)
            plt.figure(3)
            disp.specshow(mel_predictions[0].T, cmap = 'coolwarm')
            plt.colorbar()
            plt.show()'''
        '''plt.figure(3)
        plt.plot(np.abs(np.asarray(pred_before)-np.asarray(pred_after)), 'b', marker= 'o', label='orig')
        #plt.plot(pred_after, 'r', marker= 'o', label='masked')
        plt.legend()
        plt.ylim(0, 1)
        plt.grid()
        plt.show()'''
    ground = (np.asarray(pred_before))>0.66         # threshold comes from Jan's code
    ground_masked = (np.asarray(pred_after))>0.66
    count_pass = 0
    count_fail = 0
    for i in range(len(ground)):
        print(pred_before[i], pred_after[i])
        print(ground[i], ground_masked[i])
        if ground[i]==ground_masked[i]:
            count_pass +=1
        else:
            count_fail +=1
    print("Total instances:%d"%(count_pass+ count_fail))
    print("Number of fails:%d" %(count_fail))
    

if __name__ == '__main__':
    main()
