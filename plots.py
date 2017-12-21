'''
Created on 9 Dec 2017

@author: Saumitra
'''

import matplotlib.pyplot as plt
import librosa.display as disp

'''# Input (Unnormalised Mel spectrogram)
plt.figure(1)
plt.subplot(2, 3, 1)
disp.specshow(excerpts[excerpt_idx].T, y_axis='mel', hop_length= 315, x_axis='time', fmin=27.5, fmax=8000, cmap = 'coolwarm')
plt.title('Input Mel-spectrogram')
plt.xlabel('Time')
plt.ylabel('Hz')
plt.colorbar()

# Input, normalised and thresholded mel-spectrogram (top25%)
plt.subplot(2, 3, 2)
disp.specshow(((util.normalise(excerpts[excerpt_idx]))>0.75).T, y_axis='mel', hop_length= 315, x_axis='time', fmin=27.5, fmax=8000, cmap = 'gray_r')
plt.title('Input Mel-spectrogram (N_T_top25%)')
plt.xlabel('Time')
plt.ylabel('Hz')
plt.colorbar()

# Input, normalised and thresholded mel-spectrogram (top35%)
plt.subplot(2, 3, 3)
disp.specshow(((util.normalise(excerpts[excerpt_idx]))>0.65).T, y_axis='mel', hop_length= 315, x_axis='time', fmin=27.5, fmax=8000, cmap = 'gray_r')
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
disp.specshow(((util.normalise(mel_predictions[0]))>0.75).T, y_axis='mel', hop_length= 315, x_axis='time', fmin=27.5, fmax=8000, cmap='gray_r')
plt.xlabel('Time')
plt.ylabel('Hz')
plt.colorbar()

# Input, normalised and thresholded mel-spectrogram (top 35%)        
plt.subplot(2, 3, 6)
plt.title('Inversion from FC8((N_T_top35%))')
disp.specshow(((util.normalise(mel_predictions[0]))>0.65).T, y_axis='mel', hop_length= 315, x_axis='time', fmin=27.5, fmax=8000, cmap='gray_r')
plt.xlabel('Time')
plt.ylabel('Hz')
plt.colorbar()
plt.suptitle('Plots for excerpt index: %d' %(excerpt_idx))'''

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

def plot_figures(input_excerpt, inv, mask, masked_input, excerpt_idx, flag):

    if flag:
        plt.subplot(4, 1, 1)
        disp.specshow(input_excerpt.T, y_axis='mel', hop_length= 315, x_axis='off', fmin=27.5, fmax=8000, cmap = 'coolwarm')
        plt.title('Input Mel-spectrogram')
        plt.xlabel('Time')
        plt.ylabel('Hz')
        plt.colorbar()
        
        plt.subplot(4, 1, 2)
        plt.title('Inversion from FC8')
        disp.specshow(inv.T, y_axis='mel', hop_length= 315, x_axis='off', fmin=27.5, fmax=8000, cmap='coolwarm')
        plt.xlabel('Time')
        plt.ylabel('Hz')
        plt.colorbar()
        
        plt.subplot(4, 1, 3)
        disp.specshow(mask.T, y_axis='mel', hop_length= 315, x_axis='off', fmin=27.5, fmax=8000, cmap = 'gray_r')
        plt.title('Mask based on normalised and thresholded inverted mel')
        plt.colorbar()
        
        plt.subplot(4, 1, 4)
        disp.specshow((masked_input).T, y_axis='mel', hop_length= 315, x_axis='time', fmin=27.5, fmax=8000, cmap = 'coolwarm')
        plt.colorbar()
        plt.title('Masked input Mel spectrogram')
        plt.suptitle('Plots for excerpt index: %d' %(excerpt_idx))
        plt.show()
        
        #figure_name = args.generatorfile.rsplit('/', 2)
        #plt.savefig(figure_name[0]+'/'+figure_name[1]+'/'+figure_name[1]+'_ii100.pdf', dpi = 300)
