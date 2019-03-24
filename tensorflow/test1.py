# -*- coding: utf-8 -*-

import sys
import tensorflow as tf
import numpy as np

qid = np.array([
              [104],
              [114],
              [104],
              [104],
            ])
c = qid - np.transpose(qid)
mask1 = tf.equal(c, 0)
mask1 = tf.cast(mask1, tf.float32)

n = tf.shape(mask1)[0]
mask2 = tf.ones([n,n]) - tf.diag(tf.ones(n))
mask = mask1 * mask2
num_pairs = tf.reduce_sum(mask)
sys.exit()

label = np.array([
              [2],
              [1],
              [4],
              [1],
            ])
print "label.shape"
print label.shape
batch_size = label.shape[0]
sorted_label = np.sort(label, axis=None)[::-1].reshape(-1, 1)
rel = 2 ** label - 1
sorted_rel = 2 ** sorted_label - 1
print "rel"
print rel
print "sorted_rel"
print sorted_rel

sess = tf.Session()
index = tf.reshape(tf.range(1., tf.cast(batch_size, dtype=tf.float32) + 1), tf.shape(label))
#cg_discount = tf.log(1.+index)
init_cg_discount = np.array([
                        [1.],
                        [2.],
                        [4.],
                        [5.]])
cg_discount = tf.constant(init_cg_discount)
dcg_m = rel / cg_discount
dcg = tf.reduce_sum(dcg_m)
print "cg_discount"
print sess.run(cg_discount)
print "dcg_m"
print sess.run(dcg_m)
print "dcg"
print sess.run(dcg)

# stale_ij表示a[i]排在第i位时的收益
stale_ij = tf.tile(dcg_m, [1, batch_size])
print "stale_ij"
print sess.run(stale_ij)
# new_ij表示a[i]排在第j位时的收益
new_ij = rel/tf.transpose(cg_discount)
print "new_ij"
print sess.run(new_ij)

# new_ij - stale_ij表示a[i]排在第i位和第j位时的收益变化
print "new_ij - stale_ij"
print sess.run(new_ij - stale_ij)
print ""

# stale_ji表示a[j]排在第j位时的收益
stale_ji = tf.transpose(stale_ij)
print "stale_ji"
print sess.run(stale_ji)
# new_ji表示a[j]排在第i位时的收益
new_ji = tf.transpose(new_ij)
print "new_ji"
print sess.run(new_ji)
# new_ji - stale_ji表示a[j]排在第i位和第j位时的收益变化
print "new_ji - stale_ji"
print sess.run(new_ji - stale_ji)
print ""

# dcg_delta表示a[i]和a[j]交换时dcg的变化
dcg_delta = new_ij - stale_ij + new_ji - stale_ji
print "dcg_delta"
print sess.run(dcg_delta)
dcg_max = tf.reduce_sum(sorted_rel / cg_discount)
print "dcg_max"
print sess.run(dcg_max)
ndcg_delta = tf.abs(dcg_delta) / dcg_max
print "ndcg_delta"
print sess.run(ndcg_delta)
