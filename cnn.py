import torch
from torch import nn, optim
import torch.nn.functional as F
import config as c
import filters
import performance
from dataload import EpilepsyDataset, DataSplit, channels
import numpy as np

# DEVICE CONFIGURATION:
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# DETERMINE THE REDUCED LEARNING RATES: 
lr_reduced2 = c.learning_rate*c.lr_reduction_factor2 # final learning rate
lr_reduction_factor1 = (1.0-c.lr_reduction_factor2)*0.5 + c.lr_reduction_factor2 
lr_reduced1 = c.learning_rate*lr_reduction_factor1 # Intermediate learning rate, starts when epochs == patience

# INITIALIZE DATASET:
dataset = EpilepsyDataset()

# SHUFFLE THE TRAININGSET AND CREATE BATCHES:
split = DataSplit(dataset, c.nr_of_testsubjects_per_group, c.nr_of_valsubjects_per_group, c.testsetnumber, shuffle=False) #set shuffle=false to only shuffle the train+valset, not the testset; useful for k-fold crossvalidation
train_loader, val_loader, test_loader = split.get_split()

# DEFINE PYTORCH NEURAL NETWORK MODEL:
class Net(nn.Module):
    def __init__(self):
        """Defines the neural network layers."""
        super(Net,self).__init__()
        # define convolution layer:
        self.conv1=nn.Conv2d(in_channels = channels , out_channels = c.conv1_out_channels, kernel_size = c.m_size//2, stride = c.m_size//2, padding = 0)
        
        # define linear layers:
        # note that in_features = length of flattened output from previous layer
        if c.fc1nodes > 0:
          self.fc1 = nn.Linear(in_features = c.conv1_out_channels*2*2, out_features = c.fc1nodes)
          fc_bef_last_nodes = c.fc1nodes
        if c.fc2nodes > 0:
          self.fc2 = nn.Linear(in_features = c.fc1nodes, out_features = c.fc2nodes)
          fc_bef_last_nodes = c.fc2nodes

        # define the final fc layer:
        if c.loss_function == 'crossentr':
          self.fc_final = nn.Linear(in_features = fc_bef_last_nodes, out_features = c.num_classes) # out features should be 2
        elif c.loss_function == 'bcelogits':
          self.fc_final = nn.Linear(in_features = fc_bef_last_nodes, out_features = 1) # only 1 neuron in last layer

        # define dropouts:
        self.dropoutconv = nn.Dropout(c.dropout_conv) 
        self.dropoutlinear = nn.Dropout(c.dropout_linear)

        torch.manual_seed(23)
        
    def forward(self,x): 
        """Sends input x through the neural network layers."""
        x = x
        # conv layer:
        x = self.conv1(x)
        x = self.dropoutconv(x)        
        if c.loss_function == 'crossentr': 
            x = F.relu(x) # activation function
        elif c.loss_function == 'bcelogits': 
            x = F.relu(x) # activation function

        # flatten the outputs of the CNN layer for input in FC layer:
        x = x.view(-1, c.conv1_out_channels*2*2)   

        # FC layer 1:
        if c.fc1nodes > 0:
          x = self.fc1(x)              
          if c.loss_function == 'crossentr': 
              x = F.relu(x) # activation function 
          elif c.loss_function == 'bcelogits': 
              x = F.relu(x) # activation function     
          x = self.dropoutlinear(x) 
        
        # FC layer 2:
        if c.fc2nodes > 0:
          x = self.fc2(x)       
          if c.loss_function == 'crossentr': 
              x = F.relu(x) # activation function 
          elif c.loss_function == 'bcelogits': 
              x = F.relu(x) # activation function       
          x = self.dropoutlinear(x)
        
        # final layer:
        x = self.fc_final(x)        
        if c.loss_function == 'crossentr': # no activation function at final layer
            x = x   # 
        elif c.loss_function == 'bcelogits': # sigmoid activation function at final layer
            x = torch.sigmoid(x) 
        x = self.dropoutlinear(x)
        return x
               
    def predict(self,x): # 
        """Takes an input x and predicts its class: 0 or 1."""
        ans = []  
        if c.loss_function == 'crossentr': 
            pred = F.softmax(self.forward(x), dim=1) # applies softmax to output
            for t in pred: 
                if t[0] > t[1]: # compares the 2 outputs of last FC layer
                    ans.append(0) 
                else:
                    ans.append(1)                
            return torch.tensor(ans)
        
        elif c.loss_function == 'bcelogits': 
            pred = self.forward(x) # no activation function                  
            for t in pred: # uses output of only 1 neuron in last FC layer
                if t[0] >= 0.5:
                    ans.append(1)
                else:
                    ans.append(0)                
            return torch.tensor(ans)

# DEFINE EARLYSTOPPING CLASS
class EarlyStopping:
    """Early stops the training if val loss hasn't improved after patience."""
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
    def __call__(self, val_loss, model):
        """Performs checks to decide whether to stop or continue."""
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            #print(f'EARLYSTOPPING COUNTER: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), 'checkpoint.pt')
        self.val_loss_min = val_loss


# DEFINE THE TRAINING PROCEDURE
def train_model(model, batch_size, patience, num_epochs):
    """Trains the neural network."""
    # create lists for tracking losses as the model trains:
    train_losses = []
    valid_losses = []
    test_losses = []   
    # create lists for tracking avg losses per epoch (as the model trains):
    avg_train_losses = []
    avg_valid_losses = [] 
    avg_test_losses = []       
    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=c.patience, verbose=False, delta=c.early_stop_delta)   
    global learning_rate
    # train the model:    
    for epoch in range(c.num_epochs):
        model.train() # prep model for training
        # stepwise reduction of the learning rate:
        if epoch >= 2*patience:
          learning_rate = lr_reduced2
        elif epoch >= patience:
          learning_rate = lr_reduced1
        elif epoch < patience:
          learning_rate = c.learning_rate
        for i, data in enumerate(train_loader, 1): 
            inputs, labels = data
            label = labels[:,:,0] # the class of this subject: 0 or 1
            inputs = inputs.view(inputs.size(0),channels,c.m_size,c.m_size) # inputs is a tensor with GC matrix(es); inputs.size(0)= nr of subjects in 1 batch
            optimizer.zero_grad() # clears the gradients of all optimized variables            
            # forward pass: computes predicted outputs:
            y_pred = model(inputs) 
            # calculate the loss:
            if c.loss_function == 'crossentr':
              loss = criterion(y_pred, torch.max(label, 1)[0].long()) 
              # note: torch.max returns 2 tensors: one with greatest item and one with that items index
            elif c.loss_function == 'bcelogits': 
              loss = criterion(y_pred, label) 
              # note: if bceloss: 1st arg=input=sigmoid(max(two network_output_vals)); 2nd argument=target=final output you are trying to predict = 1.0 or 0.0
            # backward pass: compute gradient of the loss w.r.t. model parameters           
            loss.backward()
            # optimization step (parameter update):
            optimizer.step()
            # document the training loss:
            train_losses.append(loss.item())
            # print(f'Epoch {epoch} | Batch: {i} | Loss: {loss.item():.4f}')
   
        # validate the model:
        for valdata, valtargets in val_loader:
            # forward pass: compute predicted outputs by passing inputs to the model
            data1 = valdata.view(valdata.size(0),channels,c.m_size,c.m_size)
            output1 = model(data1)           
            # calculate the loss:
            if c.loss_function == 'crossentr':            
              target1 = valtargets[:,0,0].long() 
            elif c.loss_function == ('bcelogits'):
              target1 = valtargets[:,:,0]
            loss1 = criterion(output1, target1) 
            # document validation loss:
            valid_losses.append(loss1.item())
        
        for testdata, testtargets in test_loader: 
            # forward pass: compute predicted outputs by passing inputs to the model
            data2 = testdata.view(testdata.size(0),channels,c.m_size,c.m_size)
            output2 = model(data2)
            # calculate the loss:
            if c.loss_function == 'crossentr':
              target2 = testtargets[:,0,0].long() 
            elif c.loss_function == ('bcelogits'):
              target2 = testtargets[:,:,0] 
            loss2 = criterion(output2,target2)
            test_losses.append(loss2.item())      

        train_loss = np.average(train_losses)   
        valid_loss = np.average(valid_losses)
        test_loss = np.average(test_losses) 
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)
        avg_test_losses.append(test_loss)   

        epoch_len = len(str(num_epochs))
        
        print_msg = (f'[{epoch:>{epoch_len}}/{num_epochs:>{epoch_len}}] ' +
                     f'train_loss: {train_loss:.5f} ' +
                     f'valid_loss: {valid_loss:.5f}' +
                     f'test_loss: {test_loss:.5f}')         
              
        # clear lists for next epoch:
        train_losses = []
        valid_losses = []
        test_losses = []
        
        # early_stopping checks if validation loss has decreased:        
        early_stopping(valid_loss, model) # if decreased then makes checkpoint of current model
        if early_stopping.early_stop: # if not decreased > epochs, then stop        
            break
      
    # load the best model (of last checkpoint):
    model.load_state_dict(torch.load('checkpoint.pt'))
    return  model, avg_train_losses, avg_valid_losses, avg_test_losses 
    

# DEFINE FUNCTION THAT DETERMINES THE ACCURACY OF THE MODEL ON THE TEST DATA
def evaluate(data_loader):
    """Determines the accuracy of the trained model on the test data."""
    model.eval()
    correct = 0 # number of correct predictions
    TruePos = 0 # number of epileptic subjects with correct diagnosis
    FalsePos = 0 # number of non-epileptic subjects with incorrect diagnosis 
    TrueNeg = 0 # number of non-epileptic subjects with correct diagnosis 
    FalseNeg = 0 # number of epileptic subjects with incorrect diagnosis 
    predictionlist = []
    labelslist = []
    for i, data in enumerate(data_loader): 
        inputs, labels = data
        labels = labels[:,0] # epilepsy or not (i.e. 1 or 0)
        inputs = inputs.view(inputs.size(0),channels,c.m_size,c.m_size)
        # forward pass:
        prediction = model.predict(inputs)
        # add the subject's prediction to a list of predictions:
        prediction = prediction.tolist()
        predictionlist = predictionlist + prediction 
        # add the subject's actual label to a list of actual labels:
        labels = labels[:,0].tolist()
        labels = [int(x) for x in labels]
        labelslist = labelslist + labels
    
    for i in range(len(predictionlist)): # compare predictions with actual labels
        if predictionlist[i] == labelslist[i] :
            correct += 1
        if predictionlist[i] == 1 and labelslist[i] == 1:
            TruePos += 1
        if predictionlist[i] == 1 and labelslist[i] == 0:
            FalsePos += 1
        if predictionlist[i] == 0 and labelslist[i] == 0:                
            TrueNeg += 1
        if predictionlist[i] == 0 and labelslist[i] == 1:
            FalseNeg += 1      
            
    print("Accuracy: {}/{} ({:.0f}%)\n".format(
        correct, len(predictionlist),
        100. * correct / len(predictionlist)))
    return round(correct/len(predictionlist),2), len(predictionlist), predictionlist



# EXECUTE THE CNN TRAINING PROCESS:

# Delete the current model:
if 'model' in locals(): 
    del model 

# Initialize the model:       
model = Net().to(device)

# Define loss criterion (cross_entropy or bce_logits):
if c.loss_function == 'crossentr':
  criterion = nn.CrossEntropyLoss()
if c.loss_function == 'bcelogits': 
  criterion = nn.BCEWithLogitsLoss()

# Define the optimizer:
if c.optimizer_set == 'SGD':
  optimizer = optim.SGD(model.parameters(), lr=c.learning_rate, momentum=c.momentum, weight_decay=c.weight_decay)
if c.optimizer_set == 'ADAM':
  optimizer = torch.optim.Adam(model.parameters(), amsgrad=False)
if c.optimizer_set == 'ADAMW':
  optimizer = torch.optim.AdamW(model.parameters(), amsgrad=False)

model = model.double()

# TRAIN THE PYTORCH MODEL ON THE DATA:
model, avg_train_losses, avg_val_losses, avg_test_losses = train_model(model, c.batch_size, c.patience, c.num_epochs)   

# Determine the trained model's performance:
trainloaderaccuracy,_,_ = evaluate(train_loader)
evalloaderaccuracy,_,_ = evaluate(val_loader)
testloaderaccuracy,TotalInstances, testloaderpredictionlist = evaluate(test_loader)

# Print loss curves:
performance.plot_loss_graphs(avg_train_losses, avg_val_losses, c.testsetnumber) 

# Determine the CNN filter that performed best during this run:
max_filter_info = filters.find_important_filter(c.conv1_out_channels, model.fc1.weight, model.fc2.weight, model.fc_final.weight)

# Determine the freq band on which the filter performed best:
max_filter_nr = max_filter_info[0]
max_filter_freqband = max_filter_info[1]
print('Filter ', max_filter_nr, ' works best on freqband: ', max_filter_freqband)

# Visualize and save the image of the best performing filter:
model_filters = model.conv1.weight.data.cpu().numpy() # creates a numpy file with the filter weights
filters.write_max_filterdata(model_filters,max_filter_freqband, max_filter_nr, c.testsetnumber-1)  
filters.show_best_filter(model_filters,1,1,max_filter_nr, max_filter_freqband, c.testsetnumber)

# Write the prediction results for this run to a csv file:
row = [testloaderaccuracy, testloaderpredictionlist]
performance.write_acc_to_csv(row)

# Copy losses of this run to a file that contains losses of all runs:
performance.copy_losses(avg_train_losses, avg_val_losses, c.patience)





