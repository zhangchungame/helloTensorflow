#coding:UTF-8


import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#载入数据集
mnist = input_data.read_data_sets("MNIST_data",one_hot=True)
#每个批次的大小
batch_size = 100
#计算一共有多少个批次
n_batch = mnist.train.num_examples // batch_size

inputImage=tf.placeholder(tf.float32,[None,784])
inputText=tf.placeholder(tf.float32,[None,10])

W=tf.Variable(tf.random_normal([784, 10]))
b=tf.Variable(tf.zeros([10]))

out=tf.nn.softmax(tf.matmul(inputImage,W)+b)

loss = tf.reduce_mean(tf.square(inputText-out))

train=tf.train.GradientDescentOptimizer(0.2).minimize(loss)

correct_prediction = tf.equal(tf.argmax(inputText,1),tf.argmax(out,1))#argmax返回一维张量中最大的值所在的位置
#求准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(1000):
        for batch in range(n_batch):
            batch_xs,batch_ys =  mnist.train.next_batch(batch_size)
            loss_res,_=sess.run([loss,train],feed_dict={inputImage:batch_xs,inputText:batch_ys})

        acc = sess.run(accuracy,feed_dict={inputImage:mnist.test.images,inputText:mnist.test.labels})
        print("Iter " + str(step) + ",Testing Accuracy " + str(acc))

