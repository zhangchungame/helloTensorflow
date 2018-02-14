# coding:utf-8
from train import *
import tensorflow as tf
from readData import *


IMAGE_HEIGHT = 60
IMAGE_WIDTH = 160
MAX_CAPTCHA = 4

# 文本转向量
char_set = number+calculate # 如果验证码长度小于4, '_'用来补齐
CHAR_SET_LEN = len(char_set)
def crack_captcha(captcha_image):
    output = crack_captcha_cnn()

    saver = tf.train.Saver()
    predict = tf.argmax(tf.reshape(output, [-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2)
    with tf.Session() as sess:
        saver.restore(sess, "./Model/model.ckpt")

        text_list = sess.run(predict, feed_dict={X: [captcha_image], keep_prob: 1})
        text = text_list[0].tolist()
        return text

output = crack_captcha_cnn()
predict = tf.argmax(tf.reshape(output, [-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2)

saver = tf.train.Saver()
sess=tf.Session()
saver.restore(sess, "./Model/model.ckpt")
input=getNextTest()
for i in range(200):
    input=getNextTest()
    image = input['img'].flatten() / 255
    text_list = sess.run(predict, feed_dict={X: [image], keep_prob: 1})
    text = text_list[0].tolist()
    print("正常：{} 预测: {}".format(input['code'],vec2text(text)))