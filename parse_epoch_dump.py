'''
Created on 31 Aug 2017

@author: Saumitra
'''
'''This code parses the epoch dump file and plots the training and validation
loss w.r.t. epoch index. We aim to see, how the training loss is responding
to the increase in epoch index'''

import matplotlib.pyplot as plt

training_loss = []
validation_loss = []

with open('epoch_dump', 'r') as f:
    for line in f:
        if line[:11] == 'Train loss:':
            for t in line.split():
                try:
                    training_loss.append(float(t))
                except ValueError:
                    pass
        elif line[:16] == 'Validation loss:':
            for t in line.split():
                try:
                    validation_loss.append(float(t))
                except ValueError:
                    pass
        else:
            pass            

plt.plot(training_loss)
plt.plot(validation_loss)
plt.xlabel('Epoch Index')
plt.ylabel('Loss')
plt.show()