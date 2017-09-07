import numpy as np
import matplotlib.pyplot as plt

N = 100 # number of points per class
D = 2 # dimensionality
K = 3 # number of classes
X = np.zeros((N*K,D)) # data matrix (each row = single example)
y = np.zeros(N*K ,dtype='uint8') # class labels
for j in range(K):
  ix = range(N*j,N*(j+1))
  r = np.linspace(0.0,1,N) # radius
  t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 # theta
  X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
  y[ix] = j
# lets visualize the data:
#plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
#plt.show()


num_examples = X.shape[0]
alpha = 1e-3
learning_rate = 1
h = 100 #hidden layer size
W = 0.01* np.random.randn(D, h)
b = np.zeros((1, h))
W2 =  0.01* np.random.randn(h, K)
b2 = np.zeros((1, K))

for i in range(10000):
    hidden_layer = np.maximum(0, np.dot(X, W) + b)
    scores = np.dot(hidden_layer, W2) + b2
    # compute the class probabilities
    exp_scores = np.exp(scores)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) # [N x K]
    correct_probs = probs[range(num_examples),y]

    data_loss =  np.sum(-np.log(correct_probs))/num_examples
    reg_loss = 0.5* alpha* np.sum(W*W) + 0.5* alpha* np.sum(W2*W2)
    loss = data_loss + reg_loss
    if(i % 1000==0):
        print("Iteration: "+ str(i)+ ", loss:"+str(loss))


    #dL/df = pz - 1(y==z)
    dscores = probs
    dscores[range(num_examples), y]-=1
    dscores/= num_examples
    #dL/dW
    dW2 = np.dot(hidden_layer.T, dscores)
    db2 = np.sum(dscores, axis=0, keepdims=True)

    #back propagate
    dhidden = np.dot(dscores, W2.T)
    dhidden[hidden_layer < 0] =0
    dW = np.dot(X.T, dhidden)
    db = np.sum(dhidden, axis=0, keepdims=True)


    #parameter update
    dW2 += alpha*W2
    dW += alpha*W
    W2 += -learning_rate * dW2
    b2 += -learning_rate * db2
    W += -learning_rate * dW
    b += -learning_rate * db

hidden_layer = np.maximum(0, np.dot(X, W) + b)
scores = np.dot(hidden_layer, W2) + b2
exp_scores = np.exp(scores)
probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) # [N x K]

accuracy = np.mean(np.argmax(probs, axis=1) == y)* 100
print("Accuracy is "+ str(accuracy))



