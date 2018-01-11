# coding:utf-8


# Part7: 通过tf.train.ExponentialMovingAverage的variables_to_restore()函数获取变量重命名字典

import tensorflow as tf

v = tf.Variable(0, dtype=tf.float32, name="v")
# 注意此处的变量名称name一定要与已保存的变量名称一致
ema = tf.train.ExponentialMovingAverage(0.99)
print(ema.variables_to_restore())
# {'v/ExponentialMovingAverage': <tf.Variable 'v:0' shape=() dtype=float32_ref>}
# 此处的v取自上面变量v的名称name="v"

saver = tf.train.Saver(ema.variables_to_restore())

with tf.Session() as sess:
    saver.restore(sess, "./Model/model_ema.ckpt")
    print(sess.run(v)) # 0.0999999