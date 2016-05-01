import pandas as pd
import numpy as np
from helpers import *
from collections import Counter

classification_field = 'Species'
num_hidden_layer_nodes = 6
data_file_path = '../Data/iris.csv'

data = pd.read_csv(data_file_path)
data = data.iloc[np.random.permutation(len(data))]
data = data.reset_index(drop=True)

fields = [col for col in data.columns if col != classification_field]
num_classes = len(Counter(data[classification_field]))

k = 10

subset_size = len(data)/k

fold_accuracies = []
for i in range(k):
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

    initializer_scale = 0.004

    hidden_bias = np.ones((1, num_hidden_layer_nodes)) * initializer_scale
    hidden_weights = np.ones((len(fields), num_hidden_layer_nodes)) * initializer_scale

    output_bias = np.ones((1, num_classes)) * initializer_scale
    output_weights = np.ones((num_hidden_layer_nodes, num_classes)) * initializer_scale

    parameters = [hidden_weights,hidden_bias,output_weights,output_bias]

    n = 500

    #learning rate and momentum tuned by trial and error
    learning_rate = 0.0018
    momentum = .93

    #initialize parameter_updates to zeros
    parameter_updates = [np.zeros(matrix.shape) for matrix in parameters]

    for j in range(n):
        deltas = backpropogate(training_inputs, training_target, *parameters)
        parameter_updates = [momentum * update - learning_rate * delta for update,delta in zip(parameter_updates, deltas)]
        parameters = [param + update for param,update in zip(parameters, parameter_updates)]

    test_inputs = np.array(test_set.drop(classification_field,1))

    test_class_values = np.array(test_set[classification_field])

    test_target = np.zeros((len(test_set),num_classes))
    test_target[test_class_values==1,0] = 1
    test_target[test_class_values==2,1] = 1
    test_target[test_class_values==3,2] = 1

    confustion_matrix = np.zeros((3,3))

    output = np.around(neural_net(test_inputs,*parameters)[1])

    total = 0
    correct = 0

    for j in range(len(output)):
        total += 1
        prediction = output[j]
        targ = test_target[j]
        prediction_class = get_class(prediction)
        target_class = get_class(targ)

        confustion_matrix[prediction_class][target_class] += 1

        if prediction_class == target_class:
            correct += 1

    fold_accuracies.append(correct/float(total))
    print confustion_matrix

average_accuracy = sum(fold_accuracies)/len(fold_accuracies)

print '%.2f percent accuracy from %i fold cross validation' %(average_accuracy*100,k)
