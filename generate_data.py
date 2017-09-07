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

def compute_probs(X, W, b):
    scores = np.dot(X,W) + b
    exp_scores = np.exp(scores)
    probs = exp_scores/np.sum(exp_scores, axis=1, keepdims=True)
    return probs

num_examples = X.shape[0]
alpha = 1e-3
learning_rate = 1
W = 0.01* np.random.randn(D, K)
b = np.zeros((1, K))

for i in range(200):

    scores = np.dot(X, W) + b
    # compute the class probabilities
    exp_scores = np.exp(scores)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) # [N x K]


    correct_probs = probs[range(num_examples),y]
    data_loss =  np.sum(-np.log(correct_probs))/num_examples
    print(data_loss)
    reg_loss = 0.5* alpha* np.sum(W*W)
    print(reg_loss)
    loss = data_loss + reg_loss

    print("Iteration: "+ str(i)+ ", loss:"+str(loss))


    #dL/df = pz - 1(y==z)
    dscores = probs
    dscores[:, y]-=1
    dscores/= num_examples
    #dL/dW
    dW = np.dot(X.T, dscores)
    dW+= alpha*W
    db = np.sum(dscores, axis=0, keepdims=True)

    #parameter update
    W += -learning_rate * dW
    b += -learning_rate * db

probs = compute_probs(X,W,b)
accuracy = np.sum(np.argmax(probs, axis=1) == y)* 100/num_examples
print("Accuracy is "+ str(accuracy))


plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
plt.scatter(W[:,0], W[:,1])
plt.show()



