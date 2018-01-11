# coding:utf-8

# Part3: 若不希望重复定义计算图上的运算，可直接加载已经持久化的图

import tensorflow as tf

saver = tf.train.import_meta_graph("Model/model.ckpt.meta")

with tf.Session() as sess:
    saver.restore(sess, "./Model/model.ckpt") # 注意路径写法
    print(sess.run(tf.get_default_graph().get_tensor_by_name("add:0"))) # [ 3.]
