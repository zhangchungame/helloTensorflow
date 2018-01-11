# coding:utf-8

# 本文件程序为配合教材及学习进度渐进进行，请按照注释分段执行
# 执行时要注意IDE的当前工作过路径，最好每段重启控制器一次，输出结果更准确


# Part1: 通过tf.train.Saver类实现保存和载入神经网络模型

# 执行本段程序时注意当前的工作路径
import tensorflow as tf

v1 = tf.Variable(tf.constant(1.0, shape=[1]), name="v1")
v2 = tf.Variable(tf.constant(2.0, shape=[1]), name="v2")
result = v1 + v2

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.save(sess, "Model/model.ckpt")