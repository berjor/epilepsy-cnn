import os
import config as c
import filters as f
import performance as p

c.init() # initialize global parameters
c.set_hyper_param_iter_val(0) # select the 1st val of the iterated hyperparam

# ITERATE OVER THE VALS OF 1 HYPERPARAM, AS SET IN CONFIG.PY FILE
for iteration_nr in range(len(c.hyperparam_iter_vals)):  
    c.set_hyper_param_iter_val(iteration_nr) # sets iteration hyperparam value
    print('\n\n\nSTARTING CALCULATIONS FOR HYPERPARAM VAL NO: ' + str(iteration_nr+1) +': ' + c.hyperparam_to_iterate + '=' + str(c.hyperparam_iter_vals[iteration_nr]))
    os.mkdir(c.new_dir) # creates new directory for saving results   
    
    # TRAIN & TEST THE CNN NEURAL NETWORK 10 TIMES (10 RUNS):
    c.testsetnumber = 0
    for n in range(1,c.max_testsetnumber): # ranges over k-fold runs; default=range(1,11)
        c.testsetnumber += 1
        print('\nNOW STARTING RUN NUMBER: ', c.testsetnumber ,'/',c.max_testsetnumber-1)
        exec(open("cnn.py").read())
    
    # CREATE AN AVG IMAGE FROM THE 10 BEST FILTERS 
    avg_filter = f.create_avg_filter(c.best_filters)
    avg_filter_freqband = f.avg_filter_freqband(c.best_freq_bands)
    f.show_avg_filter(avg_filter, avg_filter_freqband)

    # CREATE AN AVG LOSS GRAPH FROM THE LOSS GRAPHS OF EACH RUN
    graphs = p.calculate_avg_loss_graphs(c.avg_train_losses_all_runs, c.avg_val_losses_all_runs, c.max_testsetnumber)
    train_losses_all_runs_averaged = graphs[0]
    val_losses_all_runs_averaged = graphs[1]
    p.plot_loss_graphs(train_losses_all_runs_averaged, val_losses_all_runs_averaged, 'avg_of_all_runs')

    # CALCULATE AVG ACCURACY OVER ALL 10 RUNS 
    final_acc = round(p.calculate_final_acc(), 3) # calculate acc and round off to 3 decimals
    performance.write_configs_and_final_acc_to_csv(final_acc)
    
    # RENAME RESULTS FOLDER SO THAT IT INCLUDE THE AVG ACCURACY
    new_dir_with_acc = c.new_dir[:8]+'ACC='+ str("%.2f" % final_acc) + c.new_dir[8:] # creates a folder name with 'ACC='
    print('TOTAL ACCURACY OVER ALL RUNS= ', str(final_acc))
    os.rename(str(c.new_dir),str(new_dir_with_acc)) # renames old folder name
    

  




