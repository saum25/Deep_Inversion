'''
Created on 15 Oct 2017

@author: Saumitra



'''

import numpy as np
import matplotlib.pyplot as plt

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

threshold = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.1]
fail_meannorm = [0, 5.79, 10.76, 13.99, 13.16, 13.93, 18.79, 28.88, 48.17, 59.78, 58.78]
fail_nomeannorm  = [0, 3.50, 7, 9.42, 11.31, 12.62, 18.83, 30.91, 48.50, 60.29, 59.04]
fail_meannorm_random = [0, 6.97, 12.16, 17.71, 22.64, 26.21, 30.35, 38.65, 50.80, 58.36, 58.82]
plt.plot(threshold, fail_meannorm, 'b', marker = 'o', linewidth = 2.0, label = 'normalisation')
plt.plot(threshold, fail_nomeannorm, 'r', marker = '|', mew = 2, ms = 10, linewidth = 2.0, label = 'no_normalisation')
plt.plot(threshold, fail_meannorm_random, 'g', marker = 's', linewidth = 2.0, label = 'normalisation_rand')
plt.ylabel("Prediction change (%)")
plt.xlabel("Normalisation threshold")
plt.ylim(0, 70)
plt.grid()
plt.legend()
plt.show()
