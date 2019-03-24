import sys
import tensorflow as tf
import numpy as np

x = tf.Variable(tf.truncated_normal([1]), name='x')
loss = tf.pow(x-3, 2, name='loss')

def derivative(x):
    return 2*(x-3)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)

#train_op = optimizer.minimize(loss)

grad_vars = optimizer.compute_gradients(loss)
train_op = optimizer.apply_gradients(grad_vars)

def train():
    with tf.Session() as sess:
        x.initializer.run()
        for i in range(10):
            print "x: ", x.eval()
            train_op.run()
            print "derivative"
            print derivative(x.eval())
            print "grad"
            print grad_vars[0][0].eval()
            print "updated var"
            print grad_vars[0][1].eval()
            print "loss: ", loss.eval()
            print ""

if __name__ == '__main__':
    train()
