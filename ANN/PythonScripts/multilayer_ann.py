import pandas as pd
import numpy as np
from collections import Counter
from itertools import combinations
from helpers import *

#-------SETUP VARIABLES-------------
classification_field = 'Species'
data_file_path = '../Data/iris.csv'

configuration = [4,5,4,3]
num_epochs = 300
save_network = False
#---------------------------------

#------------READ IN AND PROCESS DATA-------------
data = pd.read_csv(data_file_path)

#shuffle data
data = data.iloc[np.random.permutation(len(data))]
data = data.reset_index(drop=True)

property_fields = [col for col in data.columns if col != classification_field]

field_combinations = combinations(property_fields,len(property_fields) - configuration[0])
cols_to_drop = list(field_combinations)[0]

class_values = np.array(data[classification_field])
num_classes = len(Counter(class_values))

for col in cols_to_drop:
    data = data.drop(col,1)
#---------------------------------

#----------CROSS VALIDATION-------------
k = 10

subset_size = len(data)/k

fold_accuracies = []
confustion_matrix = np.zeros((num_classes,num_classes))
for i in range(k):
    #------------GET TEST AND TRAINING DATASETS------------
    test_start_index,test_end_index = i*subset_size, i*subset_size + subset_size
    test_set = data[test_start_index:test_end_index]
    head_data = data[0:test_start_index]
    tail_data = data[test_end_index:len(data)]
    frames = [head_data,tail_data]
    train_set = pd.concat(frames)

    training_class_values = np.array(train_set[classification_field])

    training_target = np.zeros((len(train_set),num_classes))
    training_target[training_class_values==1,0] = 1
    training_target[training_class_values==2,1] = 1
    training_target[training_class_values==3,2] = 1

    training_inputs = np.array(train_set.drop(classification_field,1))

    test_class_values = np.array(test_set[classification_field])

    test_target = np.zeros((len(test_set),num_classes))
    test_target[test_class_values==1,0] = 1
    test_target[test_class_values==2,1] = 1
    test_target[test_class_values==3,2] = 1

    test_inputs = np.array(test_set.drop(classification_field,1))
    #---------------------------------

    #---------INITIALIZE PARAMETERS-------------
    initializer_scale = 0.0005

    weight_params = []
    bias_params = []
    for j in range(len(configuration)-1):
        step_from = configuration[j]
        step_to = configuration[j+1]
        weight_params.append(np.ones((step_from,step_to)) * initializer_scale)
        bias_params.append(np.ones((1,step_to)) * initializer_scale)

    #learning rate and momentum tuned by trial and error
    learning_rate = 0.0018
    momentum = .89

    weight_updates = [(np.zeros(matrix.shape)) for matrix in weight_params]
    bias_updates = [(np.zeros(matrix.shape)) for matrix in bias_params]
    #---------------------------------

    #-----------TRAIN NETWORK----------
    error_tracker = []
    for j in range(num_epochs):
        weight_deltas,bias_deltas = backpropogate(training_inputs,training_target,weight_params,bias_params)
        weight_updates = [momentum * update - learning_rate * delta for update,delta in zip(weight_updates, weight_deltas)]
        bias_updates = [momentum * update - learning_rate * delta for update,delta in zip(bias_updates, bias_deltas)]

        weight_params = [param + update for param,update in zip(weight_params, weight_updates)]
        bias_params = [param + update for param,update in zip(bias_params, bias_updates)]

        #Commented code below populates error tracker
        '''sample_output = neural_net(test_inputs,weight_params,bias_params)[-1]
        sample_results = zip(get_predictions(sample_output),get_target_classes(test_target))
        num_errors = len([r for r in sample_results if r[0] != r[1]])
        error_tracker.append(num_errors)'''
    #---------------------------------

    #-----------TRAINED NETWORK RESULTS------------
    output = neural_net(test_inputs,weight_params,bias_params)[-1]

    results = zip(get_predictions(output),get_target_classes(test_target))
    for result in results:
        row = result[0] - 1
        col = result[1] - 1
        confustion_matrix[row][col] += 1

    num_correct = len([res for res in results if res[0] == res[1]])

    fold_accuracies.append(num_correct/float(len(results)))
    #---------------------------------
#---------------------------------

average_accuracy = sum(fold_accuracies)/float(len(fold_accuracies))

print 'Configuration: %s' %(str(configuration))
print 'Average Accuracy: %.2f percent' %(average_accuracy*100)
print 'Confusion Matrix: '
print confustion_matrix

#Used following code to output error tracker data into csv
'''with open('error.csv','w') as f:
    f.write('Epoch,Error\n')
    for i in range(len(error_tracker)):
        f.write('%i,%i\n' %(i,error_tracker[i]))'''

#------------EXTRA CREDIT OUTPUT FILE------------
if save_network:
    with open('network_parameters.txt','w') as f:
        all_params = zip(weight_params,bias_params)
        for weight,bias in all_params:
            f.write('Weights: %s\nBiases: %s\n\n' %(np.array2string(weight),np.array2string(bias)))
#---------------------------------
