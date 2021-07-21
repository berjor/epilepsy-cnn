import logging
from functools import lru_cache
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import pandas as pd
from config import testsetnumber, electrodes, norm_mode, m_size
from config import nr_of_testsubjects_per_group, nr_of_valsubjects_per_group 


# DEVICE CONFIGURATION:
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# PREPARE THE DATA:
data_folderread = Path("")

# LOAD THE APPROPRIATE EXCEL FILE THAT CONTAINS THE GC MATRICES:
if electrodes == 'FP1&F3&P3' and norm_mode == 'h_a':
  excelfiletoread = "DATA/4x3x3_60x_NOTCH_ICAFP1_SPLITx50s_MO15_FP1&F3&P3_DeltaThetaBetaGamma_DIAGDEL_NORM_ALLHALF.xls"
if electrodes == 'FP1&F3&P3' and norm_mode == 'hba':
  excelfiletoread = "DATA/4x3x3_60x_NOTCH_ICAFP1_SPLITx50s_MO15_FP1&F3&P3_DeltaThetaBetaGamma_DIAGDEL_NORM_AVGHALFPERBANDALLSUBJCTS.xls"
if electrodes == 'FP1&F3&P3' and norm_mode == 'dgc':
  excelfiletoread = "DATA/4x3x3_60x_NOTCH_ICAFP1_SPLITx50s_MO15_FP1&F3&P3_DeltaGamma_DIAGDEL_NORM_ALLHALF_CORNERED.xls"
if electrodes == 'FP1&F3&P3' and norm_mode == '___':
  excelfiletoread = "DATA/4x3x3_60x_NOTCH_ICAFP1_SPLITx50s_MO15_FP1&F3&P3_DeltaThetaBetaGamma_DIAGDEL.xls"
if electrodes == 'F&P' and norm_mode == 'h_a':
  excelfiletoread = "DATA/4x12x12_60x_NOTCH_ICAFP1_SPLITx50s_MO15_P&F_DeltaThetaBetaGamma_DIAGDEL_NORM_ALLHALF.xls"
if electrodes == 'F&P' and norm_mode == 'dg_':
  excelfiletoread = "DATA/4x12x12_60x_NOTCH_ICAFP1_SPLITx50s_MO15_P&F_DeltaGamma_DIAGDEL_NORM_ALLHALF.xls"
if electrodes == 'F&P' and norm_mode == 'hba':
  excelfiletoread = "DATA/4x12x12_60x_NOTCH_ICAFP1_SPLITx50s_MO15_P&F_DeltaThetaBetaGamma_DIAGDEL_NORM_AVGHALFPERBANDALLSUBJCTS.xls"
if electrodes == 'F&P' and norm_mode == '___':
  excelfiletoread = "DATA/4x12x12_60x_NOTCH_ICAFP1_SPLITx50s_MO15_P&F_DeltaThetaBetaGamma_DIAGDEL.xls"
if electrodes == 'F&F' and norm_mode == '___':
  excelfiletoread = "DATA/4x6x6_60x_NOTCH_ICAFP1_SPLITx50s_MO15_F&F_DeltaThetaBetaGamma_DIAGDEL.xls"
if electrodes == 'F&F' and norm_mode == 'h_u':
  excelfiletoread = "DATA/4X6X6_60x_NOTCH_ICAFP1_SPLITx50s_MO15_F&F_DeltaTheta_DIAGDEL_NORM_ALLHALF_UNDIRECT6x6.xls"
if electrodes == 'FP1&2&F3&4&P3&4' and norm_mode == 'h_a':
  excelfiletoread = "DATA/4x6x6_60x_NOTCH_ICAFP1_SPLITx50s_FP1&2&F3&4&P3&4_DeltaThetaBetaGamma_DIAGDEL_NORM_ALLHALF.xls"
if electrodes == 'FP1&2&F3&4&P3&4' and norm_mode == '___':
  excelfiletoread = "DATA/4x6x6_60x_NOTCH_ICAFP1_SPLITx50s_FP1&2&F3&4&P3&4_DeltaThetaBetaGamma_DIAGDEL.xls"
if electrodes == 'FP1&2&F3&4&P3&4' and norm_mode == 'dg_':
  excelfiletoread = "DATA/4x6x6_60x_NOTCH_ICAFP1_SPLITx50s_FP1&2&F3&4&P3&4_DeltaGamma_DIAGDEL.xls"
if electrodes == 'ALL21' and norm_mode == 'h_a':
  excelfiletoread = "DATA/4x21x21_60x_NOTCH_ICAFP1_SPLITx50s_MO15_21electrodes_DeltaThetaBetaGamma_DIAGDEL_NORM_ALLHALF.xls"
if electrodes == 'ALL21' and norm_mode == '___':
  excelfiletoread = "DATA/4x21x21_60x_NOTCH_ICAFP1_SPLITx50s_MO15_21electrodes_DeltaThetaBetaGamma_DIAGDEL.xls"
if electrodes == 'ALL21' and norm_mode == 'd2b':
  excelfiletoread = "DATA/4x21x21_60x_NOTCH_ICAFP1_SPLITx50s_MO15_21electrodes_DeltaToBeta_DIAGDEL.xls"
if electrodes == 'ALL21' and norm_mode == 'hbt':
  excelfiletoread = "DATA/4x21x21_60x_NOTCH_ICAFP1_SPLITx50s_MO15_21electrodes_DeltaThetaBetaGamma_DIAGDEL_NORM_ALLBANDSHALF_OVERTOTAL.xls"
if electrodes == 'ALL21' and norm_mode == 'dgh':
  excelfiletoread = "DATA/4x21x21_60x_NOTCH_ICAFP1_SPLITx50s_MO15_21electrodes_DeltaGamma_DIAGDEL_NORM_ALLHALF.xls"
if electrodes == 'T&F' and norm_mode == '___':
  excelfiletoread = "DATA/4x12x12_60x_NOTCH_ICAFP1_SPLITx50s_MO15_T&F_DeltaThetaBetaGamma_DIAGDEL.xls"
if electrodes == 'T&T' and norm_mode == 'd2b':
  excelfiletoread = "DATA/4X6X6_60x_NOTCH_ICAFP1_SPLITx50s_MO15_T&T_DeltaToBeta_DIAGDEL.xls"

file_to_read = data_folderread/excelfiletoread

N = 1 # number of GC matrices to load from excel for each subject (default= 1)
channels = 1 # number of channels used as input to the CNN layer (default= 1)

# Create 4-dimensional numpy array full of zeros for the 30 non-epilepsy subjects:
ConnMatrixArrayNoEpilepsy = np.zeros((30, N, channels, m_size+1, m_size)) # m_size is size of matrix 
# Create 4-dimensional numpy array full of ones for the 30 epilepsy subjects:
ConnMatrixArrayEpilepsy = np.ones((30, N, channels, m_size+1, m_size)) # m_size is size of matrix 
# Copy GC matrices from excel file to pandas:
ConnMatrices = pd.ExcelFile(file_to_read)
# Determine the number of subjects by counting excel sheets:
numberofsubjects = len(ConnMatrices.sheet_names)-1 

# Copy the GC matrices to the numpy arrays:
CountEpil = -1 
CountNoEpil = -1
for i in range(1,len(ConnMatrices.sheet_names)):
    df = ConnMatrices.parse(ConnMatrices.sheet_names[i], header=None)
    df_norm = df    # no normalization applied here
    sheetname = ConnMatrices.sheet_names[i]
    if sheetname[0] == 'N':     # if the 1st letter of the excel sheet name is "N"
        CountNoEpil = CountNoEpil+1
        for n in range(0,N):
            ConnMatrixArrayNoEpilepsy[CountNoEpil,n,0,0:m_size,0:m_size] = df_norm[n*(m_size+1):(n*(m_size+1)+m_size)]   
    if sheetname[0] == 'E':
        CountEpil = CountEpil+1
        for n in range(0,N):
            ConnMatrixArrayEpilepsy[CountEpil,n,0,0:m_size,0:m_size] = df_norm[n*(m_size+1):(n*(m_size+1)+m_size)]   
# Concatenate the arrays (separately for each group):
ConnMatrixArrayNoEpilepsyConcat = np.concatenate((ConnMatrixArrayNoEpilepsy),axis = 0) # this removes the n level of the array      
ConnMatrixArrayEpilepsyConcat = np.concatenate((ConnMatrixArrayEpilepsy),axis = 0)      

# Make a full dataset array from the 2 concatenated numpy arrays:
full_dataset = np.concatenate([ConnMatrixArrayNoEpilepsyConcat, ConnMatrixArrayEpilepsyConcat])


# INITIALIZE THE DATASET
class EpilepsyDataset(Dataset):
    """Class used for initializing the full dataset."""
    # Initialize the data
    def __init__(self):
        xy = full_dataset
        self.len = xy.shape[0]
        self.x_data = xy[:,:,0:-1] # takes for all freq bands all the rows of all matrices except the last row
        self.y_data = xy[:,0,[-1]] # only takes the row of 1 or 0's below the matrices of the first freq band
              
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

class DataSplit:
    """Class used for obtaining train-, val-, and testsets."""
    def __init__(self, dataset, nr_of_testsubjects_per_group, nr_of_valsubjects_per_group, testsetnumber, shuffle=False):
        self.dataset = dataset       
        # create a list of startindices = first timeframes of each subject
        self.startindices0 = [] # initialize list of startindices for nonepileptics
        for i in range(round(numberofsubjects*0.5)):
            self.startindices0.append(N*i)
        self.startindices1 = [] # startindices for epileptics
        for i in range(round(numberofsubjects*0.5), numberofsubjects):
            self.startindices1.append(N*i)
        # select and define the list of testset indices
        self.test_startindices0 = []
        self.test_startindices1 = []
        for i in range(nr_of_testsubjects_per_group):
            self.test_startindices0.append(self.startindices0[(testsetnumber-1)*nr_of_testsubjects_per_group+i-len(self.startindices0)])
            self.test_startindices1.append(self.startindices1[(testsetnumber-1)*nr_of_testsubjects_per_group+i-len(self.startindices1)])
        # collect all test_indices: 
        self.test_indices0 = [] 
        self.test_indices1 = []
        for r in self.test_startindices0:
            for e in range(0,N): # N=number of matrices per subject (=1)
                self.test_indices0.append(r+e)
        for r in self.test_startindices1:
            for e in range(0,N):
                self.test_indices1.append(r+e)
        # assemble the final list of testset indices:
        self.testset_indices = self.test_indices0 + self.test_indices1                
        # select and define the list of valset indices
        self.val_startindices0 = []
        self.val_startindices1 = []
        for c in range(nr_of_valsubjects_per_group):
            self.val_startindices0.append(self.startindices0[self.test_startindices0[-1]+c-self.startindices0[-1]+3])
            self.val_startindices1.append(self.startindices1[self.test_startindices1[-1]+c-self.startindices1[-1]+3])
        # collect all val_indices:
        self.val_indices0 = [] 
        self.val_indices1 = []
        for r in self.val_startindices0:
            for e in range(0,N): # N=number of matrices per subject (=1)
                self.val_indices0.append(r+e)
        for r in self.val_startindices1:
            for e in range(0,N):
                self.val_indices1.append(r+e)
        # assemble the final list of val set indices:
        self.valset_indices = self.val_indices0 + self.val_indices1
        self.valset_indices.sort()        
        # select and define the list of trainset indices:
        self.train_startindices0 = []
        self.train_startindices1 = []
        for i in self.startindices0:
            if i not in self.testset_indices and i not in self.valset_indices:
                self.train_startindices0.append(i)
        for c in self.startindices1:
            if c not in self.testset_indices and c not in self.valset_indices:
                self.train_startindices1.append(c)
        # collect all train_indices: 
        self.train_indices0 = [] 
        self.train_indices1 = []
        for r in self.train_startindices0:
            for e in range(0,N): # N=number of matrices per subject (=1)
                self.train_indices0.append(r+e)
        for r in self.train_startindices1:
            for e in range(0,N):
                self.train_indices1.append(r+e)
        # assemble the final list of train set indices:
        self.trainset_indices = []
        for i in range(len(self.train_indices0)):
            self.trainset_indices.append(self.train_indices0[i])
            self.trainset_indices.append(self.train_indices1[i])                
        # create a separate train, val and testset (because testset will not be randomized by a sampler)
        self.trainset = torch.utils.data.Subset(self.dataset, self.trainset_indices)
        self.valset = torch.utils.data.Subset(self.dataset, self.valset_indices)
        self.testset = torch.utils.data.Subset(self.dataset, self.testset_indices)
        
       
    @lru_cache(maxsize=4)
    def get_split(self, batch_size=1, num_workers=0):
        logging.debug('Initializing train-validation-test dataloaders')
        self.train_loader = self.get_train_loader(batch_size=batch_size, num_workers=num_workers)
        self.val_loader = self.get_validation_loader(batch_size=batch_size, num_workers=num_workers)
        self.test_loader = self.get_test_loader(batch_size=batch_size, num_workers=num_workers)
        return self.train_loader, self.val_loader, self.test_loader

    @lru_cache(maxsize=4)
    def get_train_loader(self, batch_size=1, num_workers=0):
        logging.debug('Initializing train dataloader')
        self.train_loader = DataLoader(self.trainset, batch_size=batch_size, sampler=None, shuffle=False, num_workers=num_workers)
        return self.train_loader

    @lru_cache(maxsize=4)
    def get_validation_loader(self, batch_size=1, num_workers=0):
        logging.debug('Initializing validation dataloader')
        self.val_loader = DataLoader(self.valset, batch_size=batch_size, sampler=None, shuffle=False, num_workers=num_workers)
        return self.val_loader

    @lru_cache(maxsize=4)
    def get_test_loader(self, batch_size=1, num_workers=0): # uses no sampler
        logging.debug('Initializing test dataloader')
        self.test_loader = DataLoader(self.testset, batch_size=batch_size, sampler=None, shuffle=False, num_workers=num_workers)
        return self.test_loader




