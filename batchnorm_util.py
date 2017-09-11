# def affine_forward(x, w, b):
#     """
#     Computes the forward pass for an affine (fully-connected) layer.
#     The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
#     examples, where each example x[i] has shape (d_1, ..., d_k). We will
#     reshape each input into a vector of dimension D = d_1 * ... * d_k, and
#     then transform it to an output vector of dimension M.
#     Inputs:
#     - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
#     - w: A numpy array of weights, of shape (D, M)
#     - b: A numpy array of biases, of shape (M,)
#     Returns a tuple of:
#     - out: output, of shape (N, M)
#     - cache: (x, w, b)
#     """
#     # dimension
#     N = x.shape[0]
#     D = np.prod(x.shape[1:])
#     x2 = np.reshape(x, (N, D))
#     out = np.dot(x2, w) + b
#     cache = (x, w, b)
#
#     return out, cache
#
#
# def affine_backward(dout, cache):
#     """
#     Computes the backward pass for an affine layer.
#     Inputs:
#     - dout: Upstream derivative, of shape (N, M)
#     - cache: Tuple of:
#       - x: Input data, of shape (N, d_1, ... d_k)
#       - w: Weights, of shape (D, M)
#     Returns a tuple of:
#     - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
#     - dw: Gradient with respect to w, of shape (D, M)
#     - db: Gradient with respect to b, of shape (M,)
#     """
#     x, w, b = cache
#     dx = np.dot(dout, w.T).reshape(x.shape)
#     dw = np.dot(x.reshape(x.shape[0], np.prod(x.shape[1:])).T, dout)
#     db = np.sum(dout, axis=0)
#
#     return dx, dw, db
#
# def batchnorm_forward(x, gamma, beta, bn_param):
#     """
#     Forward pass for batch normalization.
#     During training the sample mean and (uncorrected) sample variance are
#     computed from minibatch statistics and used to normalize the incoming data.
#     During training we also keep an exponentially decaying running mean of the mean
#     and variance of each feature, and these averages are used to normalize data
#     at test-time.
#     At each timestep we update the running averages for mean and variance using
#     an exponential decay based on the momentum parameter:
#     running_mean = momentum * running_mean + (1 - momentum) * sample_mean
#     running_var = momentum * running_var + (1 - momentum) * sample_var
#     Note that the batch normalization paper suggests a different test-time
#     behavior: they compute sample mean and variance for each feature using a
#     large number of training images rather than using a running average. For
#     this implementation we have chosen to use running averages instead since
#     they do not require an additional estimation step; the torch7 implementation
#     of batch normalization also uses running averages.
#     Input:
#     - x: Data of shape (N, D)
#     - gamma: Scale parameter of shape (D,)
#     - beta: Shift paremeter of shape (D,)
#     - bn_param: Dictionary with the following keys:
#       - mode: 'train' or 'test'; required
#       - eps: Constant for numeric stability
#       - momentum: Constant for running mean / variance.
#       - running_mean: Array of shape (D,) giving running mean of features
#       - running_var Array of shape (D,) giving running variance of features
#     Returns a tuple of:
#     - out: of shape (N, D)
#     - cache: A tuple of values needed in the backward pass
#     """
#     mode = bn_param['mode']
#     eps = bn_param.get('eps', 1e-5)
#     momentum = bn_param.get('momentum', 0.9)
#
#     N, D = x.shape
#     running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
#     running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))
#
#     out, cache = None, None
#     if mode == 'train':
#         #######################################################################
#         # TODO: Implement the training-time forward pass for batch normalization.   #
#         # Use minibatch statistics to compute the mean and variance, use these      #
#         # statistics to normalize the incoming data, and scale and shift the        #
#         # normalized data using gamma and beta.                                     #
#         #                                                                           #
#         # You should store the output in the variable out. Any intermediates that   #
#         # you need for the backward pass should be stored in the cache variable.    #
#         #                                                                           #
#         # You should also use your computed sample mean and variance together with  #
#         # the momentum variable to update the running mean and running variance,    #
#         # storing your result in the running_mean and running_var variables.        #
#         #######################################################################
#
#         # Forward pass
#         # Step 1 - shape of mu (D,)
#         mu = 1 / float(N) * np.sum(x, axis=0)
#
#         # Step 2 - shape of var (N,D)
#         xmu = x - mu
#
#         # Step 3 - shape of carre (N,D)
#         carre = xmu**2
#
#         # Step 4 - shape of var (D,)
#         var = 1 / float(N) * np.sum(carre, axis=0)
#
#         # Step 5 - Shape sqrtvar (D,)
#         sqrtvar = np.sqrt(var + eps)
#
#         # Step 6 - Shape invvar (D,)
#         invvar = 1. / sqrtvar
#
#         # Step 7 - Shape va2 (N,D)
#         va2 = xmu * invvar
#
#         # Step 8 - Shape va3 (N,D)
#         va3 = gamma * va2
#
#         # Step 9 - Shape out (N,D)
#         out = va3 + beta
#
#         running_mean = momentum * running_mean + (1.0 - momentum) * mu
#         running_var = momentum * running_var + (1.0 - momentum) * var
#
#         cache = (mu, xmu, carre, var, sqrtvar, invvar,
#                  va2, va3, gamma, beta, x, bn_param)
#     elif mode == 'test':
#         #######################################################################
#         # TODO: Implement the test-time forward pass for batch normalization. Use   #
#         # the running mean and variance to normalize the incoming data, then scale  #
#         # and shift the normalized data using gamma and beta. Store the result in   #
#         # the out variable.                                                         #
#         #######################################################################
#         mu = running_mean
#         var = running_var
#         xhat = (x - mu) / np.sqrt(var + eps)
#         out = gamma * xhat + beta
#         cache = (mu, var, gamma, beta, bn_param)
#
#     else:
#         raise ValueError('Invalid forward batchnorm mode "%s"' % mode)
#
#     # Store the updated running means back into bn_param
#     bn_param['running_mean'] = running_mean
#     bn_param['running_var'] = running_var
#
#     return out, cache
#
#
# def batchnorm_backward(dout, cache):
#     """
#     Backward pass for batch normalization.
#     For this implementation, you should write out a computation graph for
#     batch normalization on paper and propagate gradients backward through
#     intermediate nodes.
#     Inputs:
#     - dout: Upstream derivatives, of shape (N, D)
#     - cache: Variable of intermediates from batchnorm_forward.
#     Returns a tuple of:
#     - dx: Gradient with respect to inputs x, of shape (N, D)
#     - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
#     - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
#     """
#     dx, dgamma, dbeta = None, None, None
#
#     ##########################################################################
#     # TODO: Implement the backward pass for batch normalization. Store the      #
#     # results in the dx, dgamma, and dbeta variables.                           #
#     ##########################################################################
#     mu, xmu, carre, var, sqrtvar, invvar, va2, va3, gamma, beta, x, bn_param = cache
#     eps = bn_param.get('eps', 1e-5)
#     N, D = dout.shape
#
#     # Backprop Step 9
#     dva3 = dout
#     dbeta = np.sum(dout, axis=0)
#
#     # Backprop step 8
#     dva2 = gamma * dva3
#     dgamma = np.sum(va2 * dva3, axis=0)
#
#     # Backprop step 7
#     dxmu = invvar * dva2
#     dinvvar = np.sum(xmu * dva2, axis=0)
#
#     # Backprop step 6
#     dsqrtvar = -1. / (sqrtvar**2) * dinvvar
#
#     # Backprop step 5
#     dvar = 0.5 * (var + eps)**(-0.5) * dsqrtvar
#
#     # Backprop step 4
#     dcarre = 1 / float(N) * np.ones((carre.shape)) * dvar
#
#     # Backprop step 3
#     dxmu += 2 * xmu * dcarre
#
#     # Backprop step 2
#     dx = dxmu
#     dmu = - np.sum(dxmu, axis=0)
#
#     # Basckprop step 1
#     dx += 1 / float(N) * np.ones((dxmu.shape)) * dmu
#
#     return dx, dgamma, dbeta
#
#
# def batchnorm_backward_alt(dout, cache):
#     """
#     Alternative backward pass for batch normalization.
#     For this implementation you should work out the derivatives for the batch
#     normalizaton backward pass on paper and simplify as much as possible. You
#     should be able to derive a simple expression for the backward pass.
#     Note: This implementation should expect to receive the same cache variable
#     as batchnorm_backward, but might not use all of the values in the cache.
#     Inputs / outputs: Same as batchnorm_backward
#     """
#     dx, dgamma, dbeta = None, None, None
#
#     ##########################################################################
#     # TODO: Implement the backward pass for batch normalization. Store the      #
#     # results in the dx, dgamma, and dbeta variables.                           #
#     #                                                                           #
#     # After computing the gradient with respect to the centered inputs, you     #
#     # should be able to compute gradients with respect to the inputs in a       #
#     # single statement; our implementation fits on a single 80-character line.  #
#     ##########################################################################
#     mu, xmu, carre, var, sqrtvar, invvar, va2, va3, gamma, beta, x, bn_param = cache
#     eps = bn_param.get('eps', 1e-5)
#     N, D = dout.shape
#
#     dbeta = np.sum(dout, axis=0)
#     dgamma = np.sum((x - mu) * (var + eps)**(-1. / 2.) * dout, axis=0)
#     dx = (1. / N) * gamma * (var + eps)**(-1. / 2.) * (N * dout - np.sum(dout, axis=0)
#                                                        - (x - mu) * (var + eps)**(-1.0) * np.sum(dout * (x - mu), axis=0))
#
#     return dx, dgamma, dbeta
#
#
# def dropout_forward(x, dropout_param):
#     """
#     Performs the forward pass for (inverted) dropout.
#     Inputs:
#     - x: Input data, of any shape
#     - dropout_param: A dictionary with the following keys:
#       - p: Dropout parameter. We drop each neuron output with probability p.
#       - mode: 'test' or 'train'. If the mode is train, then perform dropout;
#         if the mode is test, then just return the input.
#       - seed: Seed for the random number generator. Passing seed makes this
#         function deterministic, which is needed for gradient checking but not in
#         real networks.
#     Outputs:
#     - out: Array of the same shape as x.
#     - cache: A tuple (dropout_param, mask). In training mode, mask is the dropout
#       mask that was used to multiply the input; in test mode, mask is None.
#     """
#
#     p, mode = dropout_param['p'], dropout_param['mode']
#     if 'seed' in dropout_param:
#         np.random.seed(dropout_param['seed'])
#
#     mask = None
#     out = None
#
#     if mode == 'train':
#         #######################################################################
#         # TODO: Implement the training phase forward pass for inverted dropout.   #
#         # Store the dropout mask in the mask variable.                            #
#         #######################################################################
#         mask = (np.random.rand(*x.shape) < p) / p
#         out = x * mask
#
#     elif mode == 'test':
#         ###################################################################
#         # TODO: Implement the test phase forward pass for inverted dropout.       #
#         ###################################################################
#         mask = None
#         out = x
#
#     cache = (dropout_param, mask)
#     out = out.astype(x.dtype, copy=False)
#
#     return out, cache
