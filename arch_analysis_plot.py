'''
Created on 15 Oct 2017

@author: Saumitra



'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

# code to plot recon error
'''arch = np.arange(1, 6)
nop = np.array([0.43, 0.80, 1.1, 1.5, 1.9]) # each number is multiplies by 1e5
nre_ii100=np.array([0.54, 0.49, 0.46, 0.42, 0.38]) # each number is multiplies by 1e2
nre = np.array([0.485, 0.451, 0.432, 0.422, 0.409, 0.406])
nre_final = np.array([0.403, 0.399, 0.422, 0.515, 0.552])
labels = ['Conv2 + MP3', 'Conv4', 'Conv5+ MP6', 'FC7', 'FC8']

nre_ch = [nre[i]-nre[i+1] for i in range(nre.shape[0]-1)]
#plt.plot(arch, nre_ii100, label = "nre_ii100")
#plt.plot(arch, nop)
#plt.plot(arch, nre, label = "nre")
plt.figure(figsize=(4, 1.8))
plt.plot(arch, nre_final,  'b', marker = 'o', label = "Jamendo", linewidth = 2.0)
plt.ylabel("Normalised reconstruction error", fontsize=7)
plt.xlabel("Layer inverted", fontsize=7)
plt.xticks(arch, labels, fontsize=7)
plt.ylim(0, 1)
plt.yticks(fontsize = 7)
plt.grid()'''
#plt.legend(fontsize=7)
#plt.show()
#plt.savefig('recon.pdf', dpi = 300)

# code to plot results of class change experiment
threshold = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.1]

fail_meannorm = [0, 5.79, 10.76, 13.99, 13.16, 13.93, 18.79, 28.88, 48.17, 59.78, 58.78]
fail_nomeannorm  = [0, 3.50, 7, 9.42, 11.31, 12.62, 18.83, 30.91, 48.50, 60.29, 59.04]
fail_meannorm_random = [0, 6.97, 12.16, 17.71, 22.64, 26.21, 30.35, 38.65, 50.80, 58.36, 58.82]
fail_nomeannorm_random  = [0, 5.61, 8.52, 14.40, 19.12, 23.22, 26.89, 37.79, 48.39, 58.58, 58.85]

fail_meannorm_rwc = [0, 6.22, 11.15, 14.06, 14.34, 15.15, 20.69, 35.53, 52.89, 62.40, 64.44]
fail_nomeannorm_rwc = [0, 5.33, 8.59, 10.30, 10.88, 14.10, 22.04, 39.21, 54.79, 63.20, 63.95]

area = [100, 98, 93, 86, 74, 60, 46, 30, 15, 3, 0]
area_rwc = [100, 97, 91, 83, 71, 56, 39, 23, 10, 2, 0]

#plt.plot(threshold, fail_meannorm, 'b', marker = 'o', linewidth = 2.0, label = 'normalisation')
plt.plot(threshold, fail_nomeannorm, 'r', marker = '|', mew = 2, ms = 10, linewidth = 2.0, label = 'no_normalisation')
#plt.plot(threshold, fail_meannorm_random, 'g', marker = 's', linewidth = 2.0, label = 'normalisation_rand')
#plt.plot(threshold, fail_nomeannorm_random, 'c', marker = 's', linewidth = 2.0, label = 'no_normalisation_rand')

#plt.plot(threshold, fail_meannorm_rwc, 'r', marker = 'o', linewidth = 2.0, label = 'normalisation_rwc')
plt.plot(threshold, fail_nomeannorm_rwc, 'k', marker = 'o', linewidth = 2.0, label = 'no_normalisation_rwc')

plt.plot(threshold, area, 'y', marker = 's', linewidth = 2.0, label = 'area')
plt.plot(threshold, area_rwc, 'g', marker = 's', linewidth = 2.0, label = 'area_rwc')


plt.ylabel("Prediction change or 1 - recall (%)")
plt.xlabel("Normalisation threshold")
plt.ylim(0, 100)
plt.grid()
plt.legend()
plt.show()

'''# code to plot histogram and cc analysis
gs = gridspec.GridSpec(2, 3)
ax0 = plt.subplot(gs[0, :]) # histogram
ax1 = plt.subplot(gs[1, 0])
ax2 = plt.subplot(gs[1, 1])
ax3 = plt.subplot(gs[1, 2])

idx = 5
index = 3610
abs_pred_err = []

with np.load('prederr_cc.npz') as f:
    abs_pred_err = [f['mt_%d' %i] for i in range(len(f.files))]
data = np.vstack([abs_pred_err[5][0, :index], abs_pred_err[6][0, :index], abs_pred_err[7][0, :index]]).T
bins = np.linspace(0, 1, 11)

ax0.hist(data, bins = bins, cumulative=False, alpha = 0.7, label = ['mt_0.5', 'mt_0.6', 'mt_0.7'])
ax0.set_xticks(np.arange(0, 1 + 0.1, 0.1))
ax0.set_yticks(np.arange(0, 4000+1, 250))
ax0.set_xlabel('bins')
ax0.set_ylabel('frequency')
ax0.set_title('CDF Histogram of the absolute prediction error')
ax0.legend(loc='upper left')
ax0.grid()

ax1.scatter(range(len(abs_pred_err[idx][0])), abs_pred_err[idx][0], c = abs_pred_err[idx][1].reshape(1, -1), cmap=plt.cm.cool, alpha = 0.7)
ax1.plot(0.2*np.ones(len(abs_pred_err[idx][0])), 'r--', linewidth = 2.0)
ax1.plot(0.6*np.ones(len(abs_pred_err[idx][0])), 'b--', linewidth = 2.0)
ax1.set_xlim(-100, len(abs_pred_err[idx][0])+ 100)
ax1.set_ylim(-0.1, 1.1)
ax1.set_xlabel('instance index')
ax1.set_ylabel('abs prediction error')
ax1.set_title('abs prediction error vs class change (mt_0.5)')

ax2.scatter(range(len(abs_pred_err[idx + 1][0])), abs_pred_err[idx + 1][0], c = abs_pred_err[idx + 1][1].reshape(1, -1), cmap=plt.cm.cool, alpha = 0.7)
ax2.set_xlim(-100, len(abs_pred_err[idx + 1][0])+ 100)
ax2.set_ylim(-0.1, 1.1)
ax2.plot(0.2*np.ones(len(abs_pred_err[idx + 1][0])), 'r--', linewidth = 2.0)
ax2.plot(0.6*np.ones(len(abs_pred_err[idx+1][0])), 'b--', linewidth = 2.0)
ax2.set_xlabel('instance index')
ax2.set_ylabel('abs prediction error')
ax2.set_title('abs prediction error vs class change (mt_0.6)')

ax3.scatter(range(len(abs_pred_err[idx + 2][0])), abs_pred_err[idx+2][0], c = abs_pred_err[idx+2][1].reshape(1, -1), cmap=plt.cm.cool, alpha = 0.7)
ax3.set_xlim(-100, len(abs_pred_err[idx +2][0])+ 100)
ax3.set_ylim(-0.1, 1.1)
ax3.plot(0.2*np.ones(len(abs_pred_err[idx+2][0])), 'r--', linewidth = 2.0)
ax3.plot(0.6*np.ones(len(abs_pred_err[idx+2][0])), 'b--', linewidth = 2.0)
ax3.set_xlabel('instance index')
ax3.set_ylabel('abs prediction error')
ax3.set_title('abs prediction error vs class change (mt_0.7)')

plt.show() # for whole figure'''
    
