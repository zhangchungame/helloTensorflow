# coding:utf-8


# Part6: 通过变量重命名直接读取变量的滑动平均值

import tensorflow as tf

v = tf.Variable(0, dtype=tf.float32, name="v")
saver = tf.train.Saver({"v/ExponentialMovingAverage": v})

with tf.Session() as sess:
    saver.restore(sess, "./Model/model_ema.ckpt")
    print(sess.run(v)) # 0.0999999
