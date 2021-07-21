# epilepsy-cnn

This repository contains the python code that I created for my master thesis 'Machine Learning and EEG in Epilepsy'.
A convolutional neural network (CNN) was trained on brain connectivity data for binary diagnosis of epilepsy.
The CNN consisted of one CNN layer and two fully connected (FC) layers.
The connectivity data were derived from 50 secs of resting-state surface EEG recordings, using Granger Causality (GC).
The used EEG data were recordings of 60 patients from the TUH EEG database; 30 with epilepsy, and 30 without epilepsy.
My master thesis describes these 60 patients and the EEG data preprocessing in detail.
This code was run on Google COLAB, using Python version 3.6.9 and torch version 1.9.0+cu102.




## CONTENTS OF REPOSITORY:

1. DATA: directory containing the Granger Connectivity (GC) matrices, preprocessed according to excel file name (see below for further info).

2. RESULTS: directory where the diagnostic performance results and trained filters of the CNN will be automatically stored.

3. config.py: where the user specifies the training hyperparameters and CNN architecture.

4. main.py: defines/controls the overall process.

5. dataload.py: defines how the GC matrices are loaded and transformed into a training-, validation-, and testset.

6. cnn.py: defines the CNN neural network class and executes the training process.

7. performance.py: contains functions used for saving results, such as loss curves.

8. filters.py: contains functions used for saving images of learned CNN filters.

				
				

## GETTING STARTED:
1. First copy the above folders and files, using the same relative locations.
2. Run the main.py script (in COLAB: !python3 main.py). 
The algorithm will automatically perform a k-fold cross validation (10 runs by default) and store the results in the RESULTS folder.
3. Tweak hyperparameters/training data/CNN architecture in the config.py file.


			
			

## NOTES:
Results will be automatically generated and stored as a new directory within the RESULTS directory. 
The name of this new directory contains all info regarding used hyperparameters, CNN architecture and obtained accuracies. 
For example: after running the main.py, the following (new) folder name may appear in the RESULTS directory:

	ACC=0.70_FP1&F3&P3_NRM=h_a_TST=3_VAL=6_CNV=28_DC=0_FC1=64_FC2=16_DL=0.25_EP=600_PA=80_DEL=-0.01_LR=0.001_RED2=0.333_MO=0.96_SGD_crossentr_BS=6_WD=0.01

#### Clarification of used abbreviations in this new folder name:

ACC=0.70, means that the trained CNN models had a diagnostic accuracy of 70% (on average) for predicting whether a patient has epilepsy or not.

FP1&F3&P3, are the names of the EEG electrodes that were used to generate GC matrices (F=Frontal, P=Parietal).

NRM=h_a, describes how the GC matrices were normalized/scaled.

TST=3, testset, contained 3 epileptic and 3 non-epileptic patients (suitable for 10-fold cross validation).

VAL=6, validation set, contained 6 epileptic and 6 non-epileptic patients.

CNV=28, convolution layer size, number of filters trained by the CNN layer.

DC=0, dropout CNN, means that the dropout for the CNN layer was 0.0.

FC1=64, fully connected, means that the 1st fully connected layer consisted of 64 neurons.

FC2=16, fully connected, means that the 2nd fully connected layer consisted of 16 neurons.

DL=0.25, dropout linear, means that the dropout for the FC layers was 0.25.

EP=600, epochs, means that the maximum number of training epochs was 600.

PA=80, patience (nr of epochs), for earlystopping.

DEL=-0.01, delta, for earlystopping: a value of > -0.01 change in val loss qualified as improvement.

LR=0.001, learning rate (lr), the training process started with a lr of 0.001.

RED2=0.333, lr reduction factor, after epochs=(2xpatience) training continued with lr = 33.3% of its initial value (0.000333).
Note: lr is reduced in 2 equal steps: between epochs=(1xpatience) and epochs=(2xpatience) lr has the intermediate value lr=0.000666.

MO=0.96, momentum

SGD, stochastic gradient descent, name of the used optimizer for updating the weights of the network.

crossentr, crossentropy, name of the loss function, for calculating the loss value.

BS=6, batch size, number of processed samples before network weights gets updated.

WD=0.01, weight decay, regularization that pushes the network weights towards zero.


				
				
## HOW TO TWEAK PARAMETERS:

All of the above parameters (except acc) can be manually tweaked in the config.py file.
To accomodate a more structured experimentation it is possible (in config.py) to iterate over multiple values of one hyperparameter. 

For example: to generate (sequentially) 3 different results, each obtained with a different momentum value, set the following in config.py:

hyperparam_to_iterate = 'mo' 

hyperparam_iter_vals = [0.94, 0.96, 0.99]



				
				
## NOTES ON INPUT DATA:

The excel files in the DATA folder contain GC matrices.
Each excel file contains 60 sheets, one sheet for each patient; sheet names starting with E are epileptic patients, starting with N are non-epileptic.

Excel file names describe: size of GC matrices, EEG preprocessing, electrodes, frequency bands and GC matrix normalization method.

For example:

4x3x3_60x_NOTCH_ICAFP1_SPLITx50s_MO15_FP1&F3&P3_DeltaThetaBetaGamma_DIAGDEL_NORM_ALLHALF

#### Clarification of these abbreviations:

4x3x3, means that the data in the excel file (which are used as CNN input) are 6x6 sized images, each consisting of four 3x3 sized GC matrices.

Each of these 3x3 GC matrices was obtained for another EEG freq band (delta=top left image quadrant, theta=top right, beta=bottom left and gamma=bottom right).

60x, simply means there are 60 such images, 1 image per patient/excel sheet.

NOTCH, means that the EEG recordings were preprocessed with a notch filter at 60, 120 and 180 Hz.

ICAFP1, means that the EEG recordings were preprocessed with eyeblink artifact removal, using ICA, removing the signal closest to the FP1 electrode.

SPLITx50s, means that, for each patient, only 50 secs of EEG recording was used (calmest 50sec upon visual inspection, from the first 5 minutes of the EEG session).

MO15, means that the Granger Causality connectivity measure was calculated with model order=15

FP1&F3&P3, the names of the (3) used EEG electrodes. If this is F&P, then 6 frontal + 6 parietal electrodes were used.

DeltaThetaBetaGamma, names of the freq bands for which GC matrices were derived and included in the input images.

DIAGDEL, means that the diagonals of the GC matrices were set to zero, thus connectivity of an electrode with itself was assumed zero.

NORM_ALLHALF, means that for each patient all GC values were scaled uniformly over the complete 6x6 image, such that its average GC value=0.5.




## TIPS AND QUESTIONS:
Probably the best way to understand the code structure is by reading the main.py file first.

Feel free to message me with any questions.
