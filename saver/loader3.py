# coding:utf-8


# Part4： tf.train.Saver类也支持在保存和加载时给变量重命名

import tensorflow as tf

# 声明的变量名称name与已保存的模型中的变量名称name不一致
u1 = tf.Variable(tf.constant(1.0, shape=[1]), name="other-v1")
u2 = tf.Variable(tf.constant(2.0, shape=[1]), name="other-v2")
result = u1 + u2

# 若直接生命Saver类对象，会报错变量找不到
# 使用一个字典dict重命名变量即可，{"已保存的变量的名称name": 重命名变量名}
# 原来名称name为v1的变量现在加载到变量u1（名称name为other-v1）中
saver = tf.train.Saver({"v1": u1, "v2": u2})

with tf.Session() as sess:
    saver.restore(sess, "./Model/model.ckpt")
    print(sess.run(result)) # [ 3.]
