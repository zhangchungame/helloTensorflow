# coding:utf-8

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
x_data = np.linspace(-0.5,0.5,200)[:,np.newaxis]        #200行一列的举证，取值在-0.5到0.5
noise = np.random.normal(0,0.02,x_data.shape)           #
y_data = np.square(x_data) + noise

x = tf.placeholder(tf.float32, [None, 1])
y = tf.placeholder(tf.float32,[None,1])

W = tf.Variable(tf.random_normal([1, 10]))
b = tf.Variable(tf.zeros([1,10]))
L1 = tf.nn.tanh(tf.matmul(x,W) + b)
# L1 = tf.nn.relu(tf.matmul(x,W) + b)
# L1 = tf.nn.sigmoid(tf.matmul(x,W) + b)
# L1 = tf.matmul(x,W) + b


w2=tf.Variable(tf.random_normal([10,1]))
b2 = tf.Variable(tf.zeros([1,1]))

# y=tf.nn.tanh(tf.matmul(L1,w2)+b2)
prediction=tf.matmul(L1,w2)+b2
    # prediction=tf.nn.sigmoid(tf.matmul(L1,w2)+b2)
    # y=tf.matmul(L1,w2)+b2


loss = tf.reduce_mean(tf.square(y - prediction))

train = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter('logs/',sess.graph)
    asd=sess.run(L1,feed_dict={x:x_data})
    for i in range(1000):
        sess.run(train,feed_dict={x:x_data,y:y_data})
    prediction_value = sess.run(prediction,feed_dict={x:x_data,y:y_data})

plt.scatter(x_data, y_data,c='r')
plt.plot(x_data, prediction_value,c='b')
plt.show()
