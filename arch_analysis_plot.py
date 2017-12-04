'''
Created on 15 Oct 2017

@author: Saumitra



'''

import numpy as np
import matplotlib.pyplot as plt

arch = np.arange(1, 7)
nop = np.array([0.43, 0.80, 1.1, 1.5, 1.9]) # each number is multiplies by 1e5
nre_ii100=np.array([0.54, 0.49, 0.46, 0.42, 0.38]) # each number is multiplies by 1e2
nre = np.array([0.485, 0.451, 0.432, 0.422, 0.409, 0.406])
nre_final = np.array([0.506, 0.403, 0.399, 0.422, 0.515, 0.552])
labels = ['Conv1', 'Conv2 + MP3', 'Conv4', 'Conv5+ MP6', 'FC7', 'FC8']

nre_ch = [nre[i]-nre[i+1] for i in range(nre.shape[0]-1)]
#plt.plot(arch, nre_ii100, label = "nre_ii100")
#plt.plot(arch, nop)
#plt.plot(arch, nre, label = "nre")
plt.plot(arch, nre_final,  'b', marker = 'o', label = "Jamendo", linewidth = 2.0)
plt.ylabel("Normalised reconstruction error")
plt.xlabel("Layer inverted")
plt.xticks(arch, labels)
plt.ylim(0, 1)
plt.grid()
plt.legend()
plt.show()
