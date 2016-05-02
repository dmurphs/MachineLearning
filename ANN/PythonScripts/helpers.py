import numpy as np

def sigmoid(z):
    '''Sigmoid function for neural network transfers'''
    return 1 / (1 + np.exp(-z))

def neural_net(inputs,weights,biases):
    '''Return activations for each layer in neural net'''
    activations = [inputs]
    for i in range(len(weights)):
        activations.append(sigmoid(np.dot(activations[i],weights[i]) + biases[i]))

    return activations[1:]

def error(activations,previous_error,previous_weights):
    '''Calculate activation errors'''
    return np.multiply(np.multiply(activations,(1 - activations)), previous_error.dot(previous_weights.T))

def gradient(previous_activations,layer_error):
    '''Get gradient for layer based on previous activations and error'''
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

def get_predictions(output):
    '''Get class predictions from network output'''
    return [np.argmax(row) + 1 for row in output]

def get_target_classes(target):
    '''Get target classes'''
    return [np.argmax(row) + 1 for row in target]
