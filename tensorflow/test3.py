import tensorflow as tf;  
import numpy as np;  

a = np.array([[1,2],[3,4]])
b = np.array([[5,6],[7,8]])
c = tf.concat([a,b], 0)
d = tf.concat([a,b], 1)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print sess.run(c)
    print ""
    print sess.run(d)
