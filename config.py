# USE THIS FILE TO CONFIGURE HYPERPARAMETERS AND CNN ARCHITECTURE
import os

# HYPERPARAMETERS:
num_classes = 2
num_epochs = 600 # number of epochs; 1 epoch=1 cycle through full training dataset
patience = 80 # early stopping patience; = how long to wait after last time validation loss improved
early_stop_delta = -0.01
learning_rate = 0.001
lr_reduction_factor2 = 0.333 # learning rate reduction factor, used after epochs > 2*patience
momentum= 0.96
weight_decay= 0.01
batch_size = 6
optimizer_set = 'SGD' # set optimizer to 'ADAM' or 'ADAMW' or 'SGD'
loss_function = 'crossentr' # set to 'crossentr' or 'bcelogits'. 'bce' and 'mse' not supported.

# CNN ARCHITECTURE CONFIG:
conv1_out_channels= 28  # number of filters trained by the conv layer
fc1nodes = 64
fc2nodes = 16
dropout_conv = 0 # dropout for conv layers, please set value < dropout_linear
dropout_linear = 0.25  # dropout for FC layers

# DATASET CONFIGURATIONS:
nr_of_testsubjects_per_group = 3 # a group = 30 subjects; set to 1 or 3 or 6 
nr_of_valsubjects_per_group = 6
electrodes = 'FP1&F3&P3' # select 'FP1&F3&P3', 'F&P', 'F&F', 'ALL21' or 'FP1&2&F3&4&P3&4'
norm_mode = 'h_a' # select '___' (no scaling), 'hba' (per freqband: avg=0.5), 'h_a' (over all 4 freqbands: avg=0.5)

# SELECT THE HYPERPARAMETER TO ITERATE OVER, AND ITS VALUES:
hyperparam_to_iterate = 'mo' # use/see abbreviations as below: 'lr', 'mo' or 'cc' etc. 
hyperparam_iter_vals = [0.94, 0.96]


# determine size of the sides of images that are used as input in the CNN input channel
if electrodes == 'FP1&F3&P3': # 3 electrodes = 3*3 GC matrices
  m_size = 6 # an image of four (3*3) GC matrices has sides = 6
if electrodes == 'F&F': # 6 frontal electrodes = 6*6 GC matrices
  m_size = 12 # an image of four (6*6) GC matrices has sides = 12
if electrodes == 'F&P': # 12 electrodes = 12*12 GC matrices
  m_size = 24 # an image of four (12*12) GC matrices has sides = 24

# determine number of (k-fold) runs:
if nr_of_testsubjects_per_group == 6: # i.e. 12 test subjects => 5 k-fold runs
  max_testsetnumber = 6 # should be 1 more than the number of actual runs, because it starts with 0
if nr_of_testsubjects_per_group == 3: # i.e. 6 test subjects => 10 k-fold runs
  max_testsetnumber = 11 # should be 1 more than the number of actual runs, because it starts with 0
if nr_of_testsubjects_per_group == 1: # i.e. 6 test subjects => 10 k-fold runs
  max_testsetnumber = 31 # should be 1 more than the number of actual runs, because it starts with 0


def set_hyper_param_iter_val(iteration_nr): 
    """Changes the val of the hyperparam that is iterated over."""     
    if hyperparam_to_iterate == 'lr':
        global learning_rate 
        learning_rate = hyperparam_iter_vals[iteration_nr] # sets value
    if hyperparam_to_iterate == 'cc':
        global conv1_out_channels 
        conv1_out_channels = hyperparam_iter_vals[iteration_nr] # sets value conv1_out_channels
    if hyperparam_to_iterate == 'fc1':
        global fc1nodes 
        fc1nodes = hyperparam_iter_vals[iteration_nr] # sets value  conv1_out_channels
    if hyperparam_to_iterate == 'fc2':
        global fc2nodes 
        fc2nodes = hyperparam_iter_vals[iteration_nr] # sets value  conv1_out_channels
    if hyperparam_to_iterate == 'dl':
        global dropout_linear
        dropout_linear = hyperparam_iter_vals[iteration_nr]
    if hyperparam_to_iterate == 'dc':
        global dropout_conv
        dropout_conv = hyperparam_iter_vals[iteration_nr]
    if hyperparam_to_iterate == 'wd':
        global weight_decay 
        weight_decay  = hyperparam_iter_vals[iteration_nr]  
    if hyperparam_to_iterate == 'mo':
        global momentum
        momentum  = hyperparam_iter_vals[iteration_nr]  
    if hyperparam_to_iterate == 'bs':
        global batch_size
        batch_size = hyperparam_iter_vals[iteration_nr]
    if hyperparam_to_iterate == 'pa':
        global patience
        patience = hyperparam_iter_vals[iteration_nr]    
    if hyperparam_to_iterate == 'ep':
        global num_epochs
        num_epochs = hyperparam_iter_vals[iteration_nr]
    if hyperparam_to_iterate == 'te':
        global nr_of_testsubjects_per_group
        nr_of_testsubjects_per_group = hyperparam_iter_vals[iteration_nr]
    if hyperparam_to_iterate == 'va':
        global nr_of_valsubjects_per_group
        nr_of_valsubjects_per_group = hyperparam_iter_vals[iteration_nr]
    global new_dir
    new_dir = ('RESULTS/_' + electrodes + '_NRM=' + norm_mode +'_TST='+str(nr_of_testsubjects_per_group) +'_VAL='+str(nr_of_valsubjects_per_group)+'_CNV=' + str(conv1_out_channels) + '_DC=' + str(dropout_conv) + '_FC1=' + str(fc1nodes) + '_FC2=' + str(fc2nodes) + '_DL='+ str(dropout_linear) +'_EP=' + str(num_epochs) + '_PA=' + str(patience) + '_DEL=' + str(early_stop_delta) + '_LR=' + str(learning_rate) + '_RED2=' + str(lr_reduction_factor2) +'_MO=' + str(momentum) + '_' + optimizer_set + '_' + loss_function + '_BS=' + str(batch_size) + '_WD=' + str(weight_decay))
    while os.path.exists(new_dir) == True: # make sure no folders have same name
      new_dir = (new_dir + '_')

def init(): 
    """Resets variables that are used for documentation of results."""         
    testsetnumber = 0   # i.e. the number of the initial run    
    # variables for plotting loss curves and filters:
    global avg_train_losses_all_runs # for overall avg train loss graph
    avg_train_losses_all_runs = num_epochs*[0]  # initiates list of zeros
    global avg_train_losses_temp # for one graph each per run
    avg_train_losses_temp = num_epochs*[0]
    global avg_val_losses_all_runs # for overall avg val loss graph
    avg_val_losses_all_runs = num_epochs*[0]   
    global avg_val_losses_temp 
    avg_val_losses_temp = num_epochs*[0]    
    global training_runs # for counting per epoch: the number of runs that were still training
    training_runs = num_epochs*[0]
    global training_runs_temp
    training_runs_temp = num_epochs*[0]    
    global best_filters # collection of all best performing filters
    best_filters = [[]]    
    global best_freq_bands # collection of all best freq bands (determined per run)
    best_freq_bands = []
    global my_path
    my_path = os.path.dirname(os.path.abspath(__file__)) # absolute path for when working directory moves around
    

    
    

