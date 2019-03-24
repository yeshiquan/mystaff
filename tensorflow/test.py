import sys
import tensorflow as tf
import numpy as np

a = np.array([1,0,3])
print "dim 1"
print a.shape
print ""

b = np.array([
              [1],
              [0],
              [3]
            ])
print "shape of b"
print b.shape
print b
print ""

c = np.transpose(b)
print "shape of transpose(b)"
print c.shape
print ""

c = b.reshape(1, -1)
print "shape of reshape(b)"
print c.shape
print ""

d = b - c
print "Result b-transpose(b)"
print d
print ""

print np.minimum(np.maximum(d, -1), 1)

label = np.array([[1,0,2,1,5]]).astype(np.float32)
label = label.reshape(-1, 1)
print label
S_ij = label - tf.transpose(label)
print S_ij

S_ij = tf.maximum(tf.minimum(1., S_ij), -1.)
P_ij = (1 / 2.) * (1 + S_ij)

sess = tf.Session()

print sess.run(P_ij)
