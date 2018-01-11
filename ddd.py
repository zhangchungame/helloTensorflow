# coding: utf-8

import tensorflow as tf


x=tf.random_normal([3, 3, 1, 32])

a=tf.nn.relu(x)
b=tf.random_uniform([1, 2], -1.0, 1.0)
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    # print sess.run(x)
    print sess.run(b)