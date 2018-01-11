# coding: UTF-8
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 随机生成1000个点，围绕在y=01x+0.3的直线周围
num_points = 1000
vectors_set = []
for i in range(num_points):
    x1 = np.random.normal(0.0, 0.55)
    y1 = 0.1*x1 + 0.3 + np.random.normal(0.0, 0.3)
    vectors_set.append([x1, y1])

# 生成一些样本
x_data = [v[0] for v in vectors_set]
y_data = [v[1] for v in vectors_set]

# 生成1维的w矩阵，取值是[-1,1]之间的随机数
w = tf.Variable(tf.random_uniform([1], -1, 1.0), name='w')
# 生成1维的b矩阵，初始值是0
b = tf.Variable(tf.zeros([1]), name='b')
# 经过计算得出预估值y
y = w*x_data + b

# 以预估值y和实际值用y_data之间的均方误差作为损失
loss = tf.reduce_mean(tf.square(y-y_data), name='loss')
# 采用梯度下降法来优化参数
optimizer = tf.train.GradientDescentOptimizer(0.5)
# 训练的过程就是最下化这个误差值
train = optimizer.minimize(loss, name='train')

sess = tf.Session()

init = tf.initialize_all_variables()
sess.run(init)

# 初始化的w和b是多少
print('w =', sess.run(w), 'b =', sess.run(b), 'loss =', sess.run(loss))
# 执行20次训练
for step in range(30):
    sess.run(train)
    print('w =', sess.run(w), 'b =', sess.run(b), 'loss =', sess.run(loss))

plt.scatter(x_data, y_data, c='r')
plt.plot(x_data, sess.run(w)*x_data+sess.run(b))
plt.show()