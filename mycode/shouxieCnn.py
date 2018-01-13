#coding:UTF-8

import tensorflow as tf
import time

from tensorflow.examples.tutorials.mnist import input_data

#载入数据集
mnist = input_data.read_data_sets("MNIST_data",one_hot=True)
#每个批次的大小
batch_size = 50
#计算一共有多少个批次
n_batch = mnist.train.num_examples // batch_size

w_alpha=0.01
b_alpha=0.1

x=tf.placeholder(tf.float32,[None,784])
keep_prob = tf.placeholder(tf.float32)  # dropout

# w1 = tf.random_normal([3, 3, 1, 32])
w1 = tf.Variable( w_alpha*tf.random_normal([5, 5, 1, 32]))
b1=tf.Variable(b_alpha*tf.random_normal([32]))
con1=tf.nn.bias_add(tf.nn.conv2d(tf.reshape(x,[-1,28,28,1]),w1,[1,1,1,1],padding='SAME'),b1)
con1=tf.nn.relu(con1)
con1 = tf.nn.max_pool(con1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
con1=tf.nn.dropout(con1,keep_prob)

w2=tf.Variable(w_alpha*tf.random_normal([5,5,32,64]))
b2=tf.Variable(b_alpha*tf.random_normal([64]))
con2=tf.nn.bias_add(tf.nn.conv2d(con1,w2,[1,1,1,1],padding='SAME'),b2)
con2 = tf.nn.max_pool(con2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
con2=tf.nn.dropout(con2,keep_prob)


dense=tf.reshape(con2,[-1 ,7*7*64])
w3=tf.Variable(w_alpha*tf.random_normal([7*7*64,1024]))
b3=tf.Variable(b_alpha*tf.random_normal([1024]))
dense1_1 = tf.nn.relu(tf.add(tf.matmul(dense, w3), b3))

w_out = tf.Variable(w_alpha*tf.random_normal([1024, 10]))
b_out = tf.Variable(b_alpha*tf.random_normal([10]))
out = tf.add(tf.matmul(dense1_1, w_out), b_out)

y_conv=tf.nn.softmax(out)

y_=tf.placeholder(tf.float32,[None,10])


#
# cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv)) #计算交叉熵
# train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy) #使用adam优化器来以0.0001的学习率来进行微调
# correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1)) #判断预测标签和实际标签是否匹配
# accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))
#
# sess = tf.Session() #启动创建的模型
# sess.run(tf.initialize_all_variables()) #旧版本
# #sess.run(tf.global_variables_initializer()) #初始化变量
#
# for i in range(500): #开始训练模型，循环训练5000次
#     batch = mnist.train.next_batch(50) #batch大小设置为50
#     if i % 100 == 0:
#         train_accuracy =sess.run(accuracy,feed_dict={x:batch[0], y_:batch[1], keep_prob:1.0})
#         # train_accuracy = accuracy.eval(session = sess,
#         #                                feed_dict = {x:batch[0], y_:batch[1], keep_prob:1.0})
#         print("step %d, train_accuracy %g" %(i, train_accuracy))
#     loss,_=sess.run([cross_entropy,train_step],feed_dict={x:batch[0], y_:batch[1], keep_prob:1.0})
#     print 'loss=',loss
#
#
# print("test accuracy %g" %accuracy.eval(session = sess,
#                                         feed_dict = {x:mnist.test.images, y_:mnist.test.labels,
#                                                      keep_prob:1.0})) #神经元输出保持不变的概率 keep_prob 为 1，即不变，一直保持输出
#
# end = time.clock() #计算程序结束时间


loss = -tf.reduce_sum(y_ * tf.log(y_conv)) #计算交叉熵
train = tf.train.AdamOptimizer(1e-4).minimize(loss) #使用adam优化器来以0.0001的学习率来进行微调
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1)) #判断预测标签和实际标签是否匹配
accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))

# loss = -tf.reduce_sum(y_ * tf.log(y_conv)) #计算交叉熵
# train = tf.train.AdamOptimizer(1e-4).minimize(loss)
# maxout=tf.argmax(y_conv,1)
# maxy=tf.argmax(y_,1)
# correct_pred = tf.equal(maxout, maxy)
# accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
images=mnist.test.images

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # batch_xs,batch_ys =  mnist.train.next_batch(batch_size)
    # con2_res=sess.run(con2,feed_dict={x:batch_xs,y:batch_ys,keep_prob:0.7})
    # dense_res=sess.run(dense,feed_dict={x:batch_xs,y:batch_ys,keep_prob:0.7})
    # dense1_res=sess.run(dense1_1,feed_dict={x:batch_xs,y:batch_ys,keep_prob:0.7})
    # outres=sess.run(out,feed_dict={x:batch_xs,y:batch_ys,keep_prob:0.7})
    for step in range(10):
        for batch in range(n_batch):
            batch_xs,batch_ys =  mnist.train.next_batch(batch_size)
            loss_res,_=sess.run([loss,train],feed_dict={x:batch_xs,y_:batch_ys,keep_prob:0.7})
            if batch %10==0:
                print time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())),'batch=',batch,'step=',step, 'loss=',loss_res
                print sess.run(accuracy,feed_dict={x:batch_xs,y_:batch_ys,keep_prob:1})
        testx,testy=mnist.test.next_batch(5000)
        print 'finally',sess.run(accuracy,feed_dict={x:testx,y_:testy,keep_prob:1})