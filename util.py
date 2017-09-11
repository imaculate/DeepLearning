import numpy as np
import math

def sigmoid(Z):
    A = 1/(1+np.exp(-Z))
    return A,Z

def relu(Z):
    A = np.maximum(0,Z)
    return A,Z #value  and cache

def relu_backward(dA, activation_cache):
    return np.multiply(dA, np.int64(activation_cache > 0))

def sigmoid_backward(dA, activation_cache):
    sig,_ = sigmoid(activation_cache)
    return sig*(1-sig)*dA

def dictionary_to_vector(parameters):
    """
    Roll all our parameters dictionary into a single vector satisfying our specific required shape.
    """
    keys = []
    first = True
    for key in parameters.keys():
        # flatten parameter
        new_vector = np.reshape(parameters[key], (-1,1))
        keys = keys + [key]*new_vector.shape[0]

        if first:
            theta = new_vector
        else:
            theta = np.concatenate((theta, new_vector), axis=0)
        first = False

    return theta, keys

def gradients_to_vector(grads):
    first = True
    for key in grads.keys():
        # flatten parameter
        if key.startswith("dW") or key.startswith("db"):
            new_vector = np.reshape(grads[key], (-1,1))

            if first:
                theta = new_vector
            else:
                theta = np.concatenate((theta, new_vector), axis=0)
            first = False

    return theta

def vector_to_dictionary(theta, param_keys):
    """
    Unroll all our parameters dictionary from a single vector satisfying our specific required shape.
    """
    parameters = {}
    param_unique_keys = np.unique(param_keys)
    L = len(param_unique_keys) //2
    for key in param_unique_keys:
        parameters[key] = theta[np.where(np.isin(param_keys, key))]
    num_next = 1
    for l in range(L, 0,-1):
        parameters['b' + str(l)] = np.reshape(parameters['b' + str(l)],(num_next, 1))
        parameters['W' + str(l)] = np.reshape(parameters['W' + str(l)],(num_next, -1))
        num_next = parameters['W' + str(l)].shape[1]

    return parameters

def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
    """
    Creates a list of random minibatches from (X, Y)

    Arguments:
    X -- input data, of shape (input size, number of examples)
    Y -- true "label" vector (1 for blue dot / 0 for red dot), of shape (1, number of examples)
    mini_batch_size -- size of the mini-batches, integer

    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """

    np.random.seed(seed)            # To make your "random" minibatches the same as ours
    m = X.shape[1]                  # number of training examples
    mini_batches = []

    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((1,m))

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k* mini_batch_size: (k+1)* mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k* mini_batch_size: (k+1)* mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, num_complete_minibatches* mini_batch_size: m]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches* mini_batch_size: m]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches
