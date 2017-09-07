import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x)),x

def relu(x):
    return np.maximum(0,x),x #value  and cache

def relu_backward(dA, activation_cache):
    return np.multiply(dA, np.int64(activation_cache > 0))

def sigmoid_backward(dA, activation_cache):
    sig = sigmoid(activation_cache)
    return sig*(1-sig)*dA

def dictionary_to_vector(parameters):
    """
    Roll all our parameters dictionary into a single vector satisfying our specific required shape.
    """
    keys = []
    count = 0
    for key in parameters.keys():

        # flatten parameter
        new_vector = np.reshape(parameters[key], (-1,1))
        keys = keys + [key]*new_vector.shape[0]

        if count == 0:
            theta = new_vector
        else:
            theta = np.concatenate((theta, new_vector), axis=0)
        count = count + 1

    return theta, keys

def vector_to_dictionary(theta, param_keys):
    """
    Unroll all our parameters dictionary from a single vector satisfying our specific required shape.
    """
    parameters = {}
    param_unique_keys = np.unique(param_keys)
    L = len(param_unique_keys)
    for key in param_unique_keys:
        parameters[key] = theta[np.where(np.isin(param_keys, key))]
    num_next = 1
    for l in range(L-1, 0):
        parameters['b' + str(l+1)] = parameters['b' + str(l+1)].reshape(num_next, 1)
        parameters['W' + str(l+1)] =  parameters['W' + str(l+1)].reshape(num_next, -1)
        num_next = parameters['W' + str(l+1)].shape[1]

    return parameters