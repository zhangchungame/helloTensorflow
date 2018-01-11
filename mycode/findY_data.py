# coding:utf-8

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
#参数概要
def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)#平均值
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)#标准差
        tf.summary.scalar('max', tf.reduce_max(var))#最大值
        tf.summary.scalar('min', tf.reduce_min(var))#最小值
        tf.summary.histogram('histogram', var)#直方图
# x_data = np.random.rand(200,1) * 10
x_data = np.linspace(-0.5,0.5,200)[:,np.newaxis]
# noise = 3
noise = np.random.normal(0,0.02,x_data.shape)
y_data = np.square(x_data) + noise

with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None, 1])
    y = tf.placeholder(tf.float32,[None,1])

with tf.name_scope('layer'):
    W = tf.Variable(tf.random_normal([1, 10]))
    b = tf.Variable(tf.zeros([1,10]))
    # L1 = tf.nn.tanh(tf.matmul(x,W) + b)
    # L1 = tf.nn.relu(tf.matmul(x,W) + b)
    L1 = tf.nn.sigmoid(tf.matmul(x,W) + b)
    # L1 = tf.matmul(x,W) + b

with tf.name_scope('out'):
    w2=tf.Variable(tf.random_normal([10,1]))
    b2 = tf.Variable(tf.zeros([1,1]))

    # y=tf.nn.tanh(tf.matmul(L1,w2)+b2)
    prediction=tf.matmul(L1,w2)+b2
    # prediction=tf.nn.sigmoid(tf.matmul(L1,w2)+b2)
    with tf.name_scope('prediction'):
        variable_summaries(prediction)
    # y=tf.matmul(L1,w2)+b2


with tf.name_scope('loss'):
    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.square(y - prediction))
        variable_summaries(loss)

with tf.name_scope('train'):
    train = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

#合并所有的summary
merged = tf.summary.merge_all()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter('logs/',sess.graph)
    for i in range(1000):
        summary,_ = sess.run([merged,train],feed_dict={x:x_data,y:y_data})
        writer.add_summary(summary,i)
    prediction_value = sess.run(prediction,feed_dict={x:x_data,y:y_data})

plt.scatter(x_data, y_data,c='r')
plt.plot(x_data, prediction_value,c='b')
plt.show()
