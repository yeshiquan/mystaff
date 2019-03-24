# -*- coding: utf-8 -*-

import sys
import tensorflow as tf
import numpy as np

t = np.array([5,6])

t = np.array([
              [2,9],
              [4,5]])

print t.shape
a = tf.expand_dims(t, 0)
print "------ shape of a ------"
print a.shape

b = tf.expand_dims(t, 1)
print "------ shape of b ------"
print b.shape

c = b - a
print "------ shape of b-a ------"
print c.shape

sess = tf.Session()
print "------ a ------"
print sess.run(a)
print "------ b ------"
print sess.run(b)
print "------ b-a ------"
print sess.run(c)
print sess.run(tf.rank(c))
