import pandas as pd
import numpy as np
from collections import Counter
from itertools import combinations

classification_field = 'Species'
data_file_path = '../Data/iris.csv'

configuration = [4,12,3]

data = pd.read_csv(data_file_path)

#shuffle data
data = data.iloc[np.random.permutation(len(data))]
data = data.reset_index(drop=True)

property_fields = [col for col in data.columns if col != classification_field]

field_combinations = combinations(property_fields,len(property_fields) - configuration[0])
cols_to_drop = list(field_combinations)[0]

for col in cols_to_drop:
    data = data.drop(col,1)

print data.columns

def sigmoid(z):
    '''Sigmoid function for neural network transfers'''
    return 1 / (1 + np.exp(-z))

def neural_net(inputs,weights,biases):
    activations = [inputs]
    for i in range(len(weights)):
        activations.append(sigmoid(np.dot(activations[i],weights[i]) + biases[i]))

    return activations[1:]

def error(activations,previous_error,previous_weights):
    return np.multiply(np.multiply(activations,(1 - activations)), previous_error.dot(previous_weights.T))

def gradient(previous_activations,layer_error):
    return np.dot(previous_activations.T,layer_error)

def backpropogate(inputs, target, weights, biases):
    '''Returns gradients for weights and biases'''
    activations = neural_net(inputs,weights,biases)

    output_error = activations[-1] - target

    #prepending to lists so orders match `activations` order
    errors = [output_error]
    output_previous = activations[-2]
    weight_gradients = [gradient(output_previous,output_error)]
    bias_gradients = [np.sum(output_error, axis=0, keepdims=True)]
    for i in range(len(activations)-1)[::-1]:
        activation_error = error(activations[i],errors[0],weights[i+1])
        errors.insert(0,activation_error)
        if i == 0:
            wg = gradient(inputs,errors[0])
            bg = np.sum(errors[0], axis=0, keepdims=True)
        else:
            wg = gradient(activations[i-1],errors[0])
            bg = np.sum(errors[0], axis=0, keepdims=True)
        weight_gradients.insert(0,wg)
        bias_gradients.insert(0,bg)

    return weight_gradients,bias_gradients

initializer_scale = 0.0005

weight_params = []
bias_params = []
for i in range(len(configuration)-1):
    step_from = configuration[i]
    step_to = configuration[i+1]
    weight_params.append(np.ones((step_from,step_to)) * initializer_scale)
    bias_params.append(np.ones((1,step_to)) * initializer_scale)

inputs = np.array(data.drop(classification_field,1))

class_values = np.array(data[classification_field])
num_classes = len(Counter(class_values))

target = np.zeros((len(data),num_classes))
target[class_values==1,0] = 1
target[class_values==2,1] = 1
target[class_values==3,2] = 1

n = 500
learning_rate = 0.002
momentum = .9

weight_updates = [(np.zeros(matrix.shape)) for matrix in weight_params]
bias_updates = [(np.zeros(matrix.shape)) for matrix in bias_params]

for i in range(n):
    weight_deltas,bias_deltas = backpropogate(inputs,target,weight_params,bias_params)
    weight_updates = [momentum * update - learning_rate * delta for update,delta in zip(weight_updates, weight_deltas)]
    bias_updates = [momentum * update - learning_rate * delta for update,delta in zip(bias_updates, bias_deltas)]

    weight_params = [param + update for param,update in zip(weight_params, weight_updates)]
    bias_params = [param + update for param,update in zip(bias_params, bias_updates)]

output = neural_net(inputs,weight_params,bias_params)[-1]

def get_predictions(output):
    return [np.argmax(row) + 1 for row in output]

def get_target_classes(target):
    return [np.argmax(row) + 1 for row in target]

r = zip(get_predictions(output),get_target_classes(target))
c = len([val for val in r if val[0] == val[1]])

print c,len(r)

#------------------------------------------------

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

    initializer_scale = 0.0005

    weight_params = []
    bias_params = []
    for i in range(len(configuration)-1):
        step_from = configuration[i]
        step_to = configuration[i+1]
        weight_params.append(np.ones((step_from,step_to)) * initializer_scale)
        bias_params.append(np.ones((1,step_to)) * initializer_scale)

    n = 500

    #learning rate and momentum tuned by trial and error
    learning_rate = 0.002
    momentum = .9

    weight_updates = [(np.zeros(matrix.shape)) for matrix in weight_params]
    bias_updates = [(np.zeros(matrix.shape)) for matrix in bias_params]

    for i in range(n):
        weight_deltas,bias_deltas = backpropogate(inputs,target,weight_params,bias_params)
        weight_updates = [momentum * update - learning_rate * delta for update,delta in zip(weight_updates, weight_deltas)]
        bias_updates = [momentum * update - learning_rate * delta for update,delta in zip(bias_updates, bias_deltas)]

        weight_params = [param + update for param,update in zip(weight_params, weight_updates)]
        bias_params = [param + update for param,update in zip(bias_params, bias_updates)]

    test_inputs = np.array(test_set.drop(classification_field,1))

    test_class_values = np.array(test_set[classification_field])

    test_target = np.zeros((len(test_set),num_classes))
    test_target[test_class_values==1,0] = 1
    test_target[test_class_values==2,1] = 1
    test_target[test_class_values==3,2] = 1

    output = neural_net(test_inputs,weight_params,bias_params)[-1]
    results = zip(get_predictions(output),get_target_classes(test_target))
    num_correct = len([res for res in results if res[0] == res[1]])

    fold_accuracies.append(num_correct/float(len(results)))

print fold_accuracies
