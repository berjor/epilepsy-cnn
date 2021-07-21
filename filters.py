import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import config as c
from collections import Counter


def find_important_filter(no_filters, weights_tensor_fc1, weights_tensor_fc2, weights_tensor_final):
    """Determines which filter results in max epilepsy score, and its associated freqband."""
    # find the fc2 neuron which connects to the max weight between fc2 and fc3:
    if c.loss_function == 'crossentr':
      fc2_maxnode = torch.max(weights_tensor_final,1)[1][1] # selects node with max weight
    elif c.loss_function == 'bcelogits': 
      # if criterion=bceloss: 1st arg=input=sigmoid(max(two network_output_vals)); 
      # 2nd argument=target=final output you are trying to predict = 1.0 or 0.0
      fc2_maxnode = torch.max(weights_tensor_final,1)[1] # selects node with max weight    
    # find the fc1 neuron which connects to the max weight between fc1 and fc2:
    fc1_maxnode = torch.max(weights_tensor_fc2,1)[1][fc2_maxnode] #gives the index of the max weight at fc2_maxnode
    # determine the index of the max weight between conv and fc1 that connects to that fc1 node:
    fc1_weight_index = torch.max(weights_tensor_fc1,1)[1][fc1_maxnode] # selects the index within that fc1node
    #determine the conv filter that is associated that weight (4 freqbands per index):
    max_filter_nr = fc1_weight_index//4    
    # find the freq band that belongs to this highest fc1 weight (4 freqbands per index):
    if fc1_weight_index%4 == 0:
        max_filter_freqband = 'delta'
    if fc1_weight_index%4 == 1:
        max_filter_freqband = 'theta'
    if fc1_weight_index%4 == 2:
        max_filter_freqband = 'beta'
    if fc1_weight_index%4 == 3:
        max_filter_freqband = 'gamma'    
    return max_filter_nr.item(), max_filter_freqband


def create_avg_filter(best_filters):
    """Returns the average image of multiple trained CNN filters."""
    scaled_filters = [] 
    for i in range(len(best_filters)): #copy the filters and scale each
        scaled_filters.append(best_filters[i]) # adds the best_filter to scaled_filters        
        scaled_filters[i] = scaled_filters[i]/(scaled_filters[i].max()-scaled_filters[i].min()) # scales the filter
    summed_filter = sum(scaled_filters)
    return summed_filter / len(best_filters) # returns avg of best_filters

def avg_filter_freqband(freqband_list): 
   """Determines which freqband name occurs most often in the input list.""" 
   occurence_count = Counter(freqband_list) 
   return occurence_count.most_common(1)[0][0] 
   

def show_best_filter(filters,rows,cols,filter_nr, filter_freqband, number):
    """Creates and saves an image of the specified trained CNN filter.""" 
    _ = plt.clf() # clears plt
    _ = plt.figure()
    w = np.array([1]) # color weight / ratio for creation of RGB image
    img1 = filters[filter_nr]
    img1 = np.transpose(img1, (1, 2, 0))
    img1 = img1/(img1.max()-img1.min())
    img1 = np.dot(img1,w)  
    _ = plt.imshow(img1,cmap= 'coolwarm')
    titletxt1 = ('Run ' + str(number) + ', filter ' + str(filter_nr) + ': ' + filter_freqband +' band')
    _ = plt.title(titletxt1)
    # specify axis labels:
    if c.electrodes == 'FP1&F3&P3':
      _ = plt.xlabel("FP1                  F3                 P3")
      _ = plt.ylabel("P3                   F3                FP1")
    if c.electrodes == 'F&F':
      _ = plt.xlabel("FP1      FP2       F3       F4        F7        F8", fontsize=11)
      _ = plt.ylabel("F8       F7        F4       F3       FP2       FP1", fontsize=11)
    if c.electrodes == 'FP1&2&F3&4&P3&4':
      _ = plt.xlabel("FP1      FP2       F3       F4        P3        P4", fontsize=11)
      _ = plt.ylabel("P4       P3        F4       F3       FP2       FP1", fontsize=11)
    if c.electrodes == 'F&P':
      _ = plt.xlabel("FP1 FP2  F3  F4  C3  C4  P3  P4  F7  F8  CZ  PZ", fontsize=11)
      _ = plt.ylabel("PZ  CZ   F8  F7  P3  P4  C4  C3  F4  F3  FP2 FP1", fontsize=11)
    if c.electrodes == 'ALL21':
      _ = plt.xlabel("FP1 FP2 F3  F4  C3  C4  P3  P4  O1  O2  F7  F8  T3  T4  T5  T6  FZ  CZ  PZ  T1  T2", fontsize=6.6)
      _ = plt.ylabel("T2  T1  PZ  CZ  FZ  T6  T5  T4  T3  F8  F7  O2  O1  P4  P3  C4  C3  F4  F3 FP2 FP1", fontsize=6.6)
    _ = plt.xticks([])
    _ = plt.yticks([])
    # save image of filter:
    filename = str('_best_filter_run_' + str(c.testsetnumber)) 
    _ = plt.savefig(os.path.join(c.my_path, c.new_dir, filename))
    _ = plt.clf()  

def write_max_filterdata(filters, freqband, filter_nr, index):    
    """Copies a freqband name to a list of freqband names, 
        and copies a CNN filter (matrix) to a list of filters.""" 
    # write freqband name to the list best_freq_bands:
    c.best_freq_bands.append(freqband) 
    # write the filter to the best_filters list:
    if index == 0:
        c.best_filters[0] = filters[filter_nr]
    else:
        c.best_filters.append(filters[filter_nr])
    print('wrote max_filterdata to best_filters list')


def show_avg_filter(some_filter, freqband):
    """Creates and saves an image of the specified (averaged) CNN filter."""
    _ = plt.clf() # clears plt
    _ = plt.figure()
    w = np.array([1]) # color weight / ratio for creation of RGB image
    img = some_filter
    img = np.transpose(img, (1, 2, 0))
    img = img/(img.max()-img.min())
    img = np.dot(img,w)
    _ = plt.imshow(img,cmap= 'coolwarm')
    _ = plt.xticks([])
    _ = plt.yticks([])
    # specify axis labels:
    if c.electrodes == 'FP1&F3&P3':
      _ = plt.xlabel("FP1                  F3                 P3")
      _ = plt.ylabel("P3                   F3                FP1")
    if c.electrodes == 'F&F':
      _ = plt.xlabel("FP1      FP2       F3       F4        F7        F8", fontsize=11)
      _ = plt.ylabel("F8       F7        F4       F3       FP2       FP1", fontsize=11)
    if c.electrodes == 'FP1&2&F3&4&P3&4':
      _ = plt.xlabel("FP1      FP2       F3       F4        P3        P4", fontsize=11)
      _ = plt.ylabel("P4       P3        F4       F3       FP2       FP1", fontsize=11)
    if c.electrodes == 'F&P':
      _ = plt.xlabel("FP1 FP2  F3  F4  C3  C4  P3  P4  F7  F8  CZ  PZ", fontsize=11)
      _ = plt.ylabel("PZ  CZ   F8  F7  P3  P4  C4  C3  F4  F3  FP2 FP1", fontsize=11)
    if c.electrodes == 'ALL21':
      _ = plt.xlabel("FP1 FP2 F3  F4  C3  C4  P3  P4  O1  O2  F7  F8  T3  T4  T5  T6  FZ  CZ  PZ  T1  T2", fontsize=6.6)
      _ = plt.ylabel("T2  T1  PZ  CZ  FZ  T6  T5  T4  T3  F8  F7  O2  O1  P4  P3  C4  C3  F4  F3 FP2 FP1", fontsize=6.6)
    
    titletxt = ('avg filter: ' + freqband +' band')
    _ = plt.title(titletxt)
    # save image of the averaged filter:
    filename = str('__img_avg_of_best_filters') 
    _ = plt.savefig(os.path.join(c.my_path, c.new_dir, filename))
    _ = plt.clf()  
