from deep_neural_network import *
from sklearn import datasets
from sklearn.model_selection import train_test_split

data = datasets.load_breast_cancer()

X = data.data
y = data.target
target_names = list(data.target_names)

X_train, X_test, Y_train, Y_test = train_test_split(X, y)

Y_train = np.reshape(Y_train,(-1,1))
Y_test = np.reshape(Y_test,(-1,1))

deepNN = model(X_train.T, Y_train.T)
predict(X_test.T, Y_test.T,deepNN)