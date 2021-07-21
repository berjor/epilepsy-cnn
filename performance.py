import csv
from pathlib import Path
import config as c 
import matplotlib.pyplot as plt
import os


def write_acc_to_csv(row): 
    """Writes the specified row of txt to csv file.
      And it creates new csv file if none exists yet."""
    filename = str("__testrunresults.csv")
    file_to_write = (os.path.join(c.my_path, c.new_dir, filename))
    f = open(file_to_write, 'a')
    with f:
        writer = csv.writer(f, lineterminator = '\n')
        writer.writerow(row)  


def write_configs_and_final_acc_to_csv(final_acc): 
    """Creates new csv file for documenting configs and results."""
    l1 = ('RESULT: ') 
    l2 = ('Average accuracy over all ' + str(c.max_testsetnumber-1) + ' runs = ' + str(final_acc))
    l3 = ('')
    l4 = ('CNN ARCHITECTURE: ')
    l5 = ('conv1_out_channels = ' + str(c.conv1_out_channels))
    l6 = ('dropout_conv = ' + str(c.dropout_conv))
    l7 = ('fc1nodes = ' + str(c.fc1nodes)) 
    l8 = ('dropout_linear = ' + str(c.dropout_linear))
    l9 = ('fc2nodes = ' + str(c.fc2nodes))
    l10 = ('dropout_linear = ' + str(c.dropout_linear))
    l11 = ('')
    l12 = ('INPUT DATA: ')
    l13 = ('electrodes = ' + str(c.electrodes))
    l14 = ('norm_mode = ' + str(c.norm_mode))
    l15 = ('nr_of_testsubjects_per_group = ' + str(c.nr_of_testsubjects_per_group)) 
    l16 = ('nr_of_valsubjects_per_group = ' + str(c.nr_of_valsubjects_per_group))        
    l17 = ('')
    l18 = ('HYPERPARAMETERS: ') 
    l19 = ('num_epochs = ' + str(c.num_epochs))
    l20 = ('patience = ' + str(c.patience))
    l21 = ('learning_rate = ' + str(c.learning_rate)) 
    l22 = ('momentum = ' + str(c.momentum))
    l23 = ('weight_decay = ' + str(c.weight_decay))
    l24 = ('batch_size = ' + str(c.batch_size))
    l25 = ('optimizer = ' + str(c.optimizer_set)) 
    l26 = ('loss_function = ' + str(c.loss_function))    
    filename = str("__configurations.txt")
    file_to_write = (os.path.join(c.my_path, c.new_dir, filename))
    with open(file_to_write,'w') as out:
      out.write('{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n'.format(l1,l2,l3,l4,l5,l6,l7,l8,l9,l10,l11,l12,l13,l14,l15,l16,l17,l18,l19,l20,l21,l22,l23,l24,l25,l26))  
   
def copy_losses(avg_train_losses, avg_val_losses, patience):
    """Adds losses to list (for loss graph). 
      And it indicates (with a list of ones) which epochs were training (excl patience)."""   
    # add train loss to list:
    for u in range(0,len(avg_train_losses)-patience):
        c.avg_train_losses_all_runs[u] += avg_train_losses[u] 
    # add val loss to list:       
    for u in range(0,len(avg_val_losses)-patience):
        c.avg_val_losses_all_runs[u] += avg_val_losses[u]
    # register +1 when the algorithm was still training:   
    for u in range(0,len(avg_val_losses)-patience):
        c.training_runs[u] += 1 # 1 if it was still training

    
def calculate_final_acc():
  """Returns the average accuracy of the trained models over all runs."""
  filename = str("__testrunresults.csv")
  file_to_read = (os.path.join(c.my_path, c.new_dir, filename))
  with open(file_to_read) as csvDataFile:
    sum_of_accs = 0.0
    csvReader = csv.reader(csvDataFile)
    for row in csvReader:
        print('row[0]=',row[0])
        sum_of_accs += float(row[0])
    return sum_of_accs/(c.max_testsetnumber-1)  

def calculate_avg_loss_graphs(avg_train_losses_all_runs, avg_val_losses_all_runs, max_testsetnumber):
    """Returns datapoints of the average loss curves of the trained models over all runs."""
    avg_train_losses_all_runs_averaged = []
    avg_val_losses_all_runs_averaged = []
    avg_train_losses_all_runs_divided = []
    avg_val_losses_all_runs_divided = []
    for i in range(len(avg_train_losses_all_runs)):
        train_runs_per_epoch = c.training_runs[i]
        if train_runs_per_epoch != 0:
            avg_train_losses_all_runs_divided += [avg_train_losses_all_runs[i]/(train_runs_per_epoch)]
            avg_val_losses_all_runs_divided += [avg_val_losses_all_runs[i]/(train_runs_per_epoch)]
    avg_train_losses_all_runs_averaged = avg_train_losses_all_runs_divided
    avg_val_losses_all_runs_averaged = avg_val_losses_all_runs_divided  
    return avg_train_losses_all_runs_averaged, avg_val_losses_all_runs_averaged
    
def plot_loss_graphs(train_losses, val_losses, title):
    """Creates and saves an image that contains a train loss and a val loss curve."""
    _ = plt.clf()
    filename = str('___loss_curves_' + str(title))
    _ = plt.plot(train_losses, label='train')
    _ = plt.plot(val_losses, label='val')
    _ = plt.xlabel("Epochs")
    _ = plt.ylabel("Loss")
    _ = plt.legend()    
    _ = plt.savefig(os.path.join(c.my_path, c.new_dir, filename))
    _ = plt.clf()
