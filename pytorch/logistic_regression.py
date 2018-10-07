import time
import math
import random

import pandas as pd
import sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score

class LogisticRegression(object):
    def __init__(self):
        self.lr = 0.00001
        self.max_iter = 5000

    def predict(self, x):
        wx = sum(self.w[j] * x[j] for j in xrange(len(self.w)))
        exp_wx = math.exp(wx)
        predict1 = exp_wx / (1 + exp_wx)
        predict0 = 1 / (1 + exp_wx)
        if predict1 > predict0:
            return 1
        return 0

    def train(self.features, labels):
        self.w = [0.0] * (len(features[0]) + 1)
        correct_count = 0
        time = 0

        while time < self.max_iter:
            index = random.randint(0, len(labels) - 1)
            x = list(features[index])
            x.append(1.0)
            y = labels[index]

            if y == self.predict_(x):
                correct_count += 1
                if correct_count > self.max_iter:
                    break
                continue
            time += 1
