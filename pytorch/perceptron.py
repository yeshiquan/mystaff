import sys
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

class Perceptron(object):
    def __init__(self, lr=0.01, n_iter=10, X_test=None, y_test=None):
        self.lr = lr
        self.n_iter = n_iter
	self.X_test = X_test
	self.y_test = y_test

    def fit(self, X, y):
        self.wb = np.zeros(1+X.shape[1])
        self.errors_ = []
        for it in range(self.n_iter):
            errors = 0
            for xi, yi in zip(X,y):
		predict_y = self.predict(xi)
		if yi*predict_y <=0:
		    self.wb[1:] += self.lr * yi * xi
		    self.wb[0:] += self.lr * yi
		    errors += 1
            self.errors_.append(errors)
	    if it % 10 == 0:
		#self.plot_decision_regions(title='Iter %d' % it)
		pass
        return self
	
    def net_input(self, xi):
	return np.dot(xi, self.wb[1:]) + self.wb[0]

    def predict(self, xi):
        return np.where(self.net_input(xi) <= 0.0, -1, 1)

    def plot_decision_regions(self, title='', resolution=0.02):
	X = self.X_test
	y = self.y_test
        # initialization colors map
        colors = ['red', 'blue']
        markers = ['o', 'x']
        cmap = ListedColormap(colors[:len(np.unique(y))])

	if title != '':
	    plt.title(title)

        # plot the decision regions
        x1_max, x1_min = max(X[:, 0]) + 1, min(X[:, 0]) - 1
        x2_max, x2_min = max(X[:, 1]) + 1, min(X[:, 1]) - 1
        xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                               np.arange(x2_min, x2_max, resolution))
        Z = self.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
        Z = Z.reshape(xx1.shape)
        plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
        plt.xlim(xx1.min(), xx1.max())
        plt.ylim(xx2.min(), xx2.max())

        # plot class samples
        for idx, cl in enumerate(np.unique(y)):
            plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                        alpha=0.8, c=cmap(idx),
                        marker=markers[idx], label=cl)
        plt.show()

    
iris = load_iris()
X = iris.data[:100,[1,2]]
y = iris.target[:100]
y = np.where(y == 1, 1, -1)

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3)
ppn = Perceptron(lr=0.1, n_iter=500, X_test=X_test, y_test=y_test)
ppn.fit(X_train, y_train)
ppn.plot_decision_regions(title='Train Model')
