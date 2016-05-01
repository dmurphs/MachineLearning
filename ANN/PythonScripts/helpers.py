import numpy as np
import pandas as pd

def sigmoid(z):
    '''Sigmoid function for neural network transfers'''
    return 1 / (1 + np.exp(-z))

def neural_net(inputs,hidden_weights,hidden_bias,output_weights,output_bias):
    '''Returns the activations for the hidden layer and ouput layer in neural network'''
    hidden_activations = sigmoid(np.dot(inputs,hidden_weights) + hidden_bias)
    output_activations = sigmoid(np.dot(hidden_activations,output_weights) + output_bias)
    return hidden_activations,output_activations

def backpropogate(inputs, target, hidden_weights, hidden_bias, output_weights, output_bias):
    '''Returns gradients for weights and biases'''
    hidden_activations,output_activations = neural_net(inputs,hidden_weights,hidden_bias,output_weights,output_bias)

    output_error = output_activations - target
    output_weight_gradient = np.dot(hidden_activations.T,output_error)
    output_bias_gradient = np.sum(output_error, axis=0, keepdims=True)

    hidden_error = np.multiply(np.multiply(hidden_activations,(1 - hidden_activations)), output_error.dot(output_weights.T))
    hidden_weight_gradient = np.dot(inputs.T,hidden_error)
    hidden_bias_gradient = np.sum(hidden_error, axis=0, keepdims=True)

    return [hidden_weight_gradient, hidden_bias_gradient, output_weight_gradient, output_bias_gradient]

def get_classification_accuracy(output,target):
    '''Return number of correct classifications and total number of classifications for accuracy analysis'''
    total = 0
    correct = 0

    for i in range(len(output)):
        total += 1
        prediction = output[i]
        targ = target[i]
        equal = [k for k in range(len(prediction)) if prediction[k] != targ[k]]
        if len(equal) == 0:
            correct += 1

    return correct,total

def get_class(l):
    pred_index = 0
    for i in range(len(l)):
        if l[i] == 1:
            pred_index = i

    return pred_index
