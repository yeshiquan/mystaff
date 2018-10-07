#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs

import time

class LR(object):
    def __init__(self, lr=0.01, max_iter=20):
        self.lr = lr
        self.max_iter = max_iter

    def sigmoid(self, x):
        return 1.0 / (1+np.exp(-x))

    def train(self):
        train_x = self.train_x
        train_y = self.train_y

        N, M = train_x.shape
        self.w = np.zeros((M, 1))

        for k in range(self.max_iter):
            predict_y = self.sigmoid(np.dot(train_x, self.w))
            self.w += self.lr * np.dot(train_x.T, (train_y - predict_y))
            if k % 100 == 0:
                print self.loss()
                #print self.w.T
                pass

        y_p_train = self.predict(self.train_x)
        print("train accuracy: %.2f%%" % (100 - np.mean(np.abs(y_p_train - self.train_y)) * 100))
        y_p_test = self.predict(self.test_x)
        print("test accuracy: %.2f%%" % (100 - np.mean(np.abs(y_p_test - self.test_y)) * 100))



    def loss(self):
        predict_y = self.sigmoid(np.dot(self.train_x, self.w))
        # 0.0000是为了防止 RuntimeWarning: divide by zero encountered in log
        loss_1 = self.train_y * np.log(predict_y+0.00001)
        loss_2 = (1 - self.train_y) * np.log(1-predict_y+0.00001)
        loss = loss_1 + loss_2
        return np.sum(loss)

    def predict(self, X):
        predict_y = self.sigmoid(np.dot(X, self.w))
        predict_y_labels = [1 if elem > 0.5 else 0 for elem in predict_y]
        return np.array(predict_y_labels)[:, np.newaxis]

    def draw(self, data_x, data_y):
        fig = plt.figure(figsize=(8,6))
        plt.scatter(data_x[:,0], data_x[:,1], c=data_y)
        plt.title("Dataset")
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.show()

    def load_data(self):
        data_x, data_y = make_blobs(n_samples= 1000, centers=2)
        self.draw(data_x, data_y)
        data_y = data_y[:, np.newaxis]

        self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(data_x, data_y, test_size=0.2)

if __name__ == '__main__':
    lr = LR(lr=0.001, max_iter=10000)
    lr.load_data()
    lr.train()
