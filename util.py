'''
Created on 9 Dec 2017

@author: Saumitra
'''

import numpy as np
import librosa
import os
import argparse

def dist_euclidean(x, y):
    '''t1 = (x - y)
    t2 = t1 ** 2
    t3 = np.sum(t2)
    t4 = np.sqrt(t3)'''
    return np.sqrt(np.sum((x-y)**2))

def dist_euclidean_matrix(x, y):
    return ((x-y)**2)

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
    
def normalise(x):
    """
    Normalise a vector/ matrix, in range 0 - 1
    """
    return((x-x.min())/(x.max()-x.min()))