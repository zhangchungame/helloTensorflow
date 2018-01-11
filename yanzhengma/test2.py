# coding:utf-8
from train import crack_captcha_cnn
from train import convert2gray
from train import X
from train import keep_prob
import tensorflow as tf
from gen_captcha import gen_captcha_text_and_image
from gen_captcha import number
from gen_captcha import alphabet
from gen_captcha import ALPHABET


IMAGE_HEIGHT = 60
IMAGE_WIDTH = 160
MAX_CAPTCHA = 4

# 文本转向量
char_set = number + alphabet + ALPHABET + ['_']  # 如果验证码长度小于4, '_'用来补齐
CHAR_SET_LEN = len(char_set)
def crack_captcha(captcha_image):
    # output = crack_captcha_cnn()
    w_alpha=0.01
    b_alpha=0.1
    #############

    x = tf.reshape(X, shape=[-1, 60, 160, 1])
    w_c1 = tf.Variable(w_alpha * tf.random_normal([3, 3, 1, 32]))
    b_c1 = tf.Variable(b_alpha * tf.random_normal([32]))
    conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, w_c1, strides=[1, 1, 1, 1], padding='SAME'), b_c1))
    conv1_2 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    w_c2 = tf.Variable(w_alpha * tf.random_normal([3, 3, 32, 64]))
    b_c2 = tf.Variable(b_alpha * tf.random_normal([64]))
    conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv1_2, w_c2, strides=[1, 1, 1, 1], padding='SAME'), b_c2))
    conv2_1 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    w_c3 = tf.Variable(w_alpha * tf.random_normal([3, 3, 64, 64]))
    b_c3 = tf.Variable(b_alpha * tf.random_normal([64]))
    conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv2_1, w_c3, strides=[1, 1, 1, 1], padding='SAME'), b_c3))
    conv3_1 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


    w_d = tf.Variable(w_alpha * tf.random_normal([8 * 32 * 40, 1024]))
    b_d = tf.Variable(b_alpha * tf.random_normal([1024]))
    dense = tf.reshape(conv3_1, [-1, w_d.get_shape().as_list()[0]])
    dense1_1 = tf.nn.relu(tf.add(tf.matmul(dense, w_d), b_d))

    w_out = tf.Variable(w_alpha * tf.random_normal([1024, MAX_CAPTCHA * CHAR_SET_LEN]))
    b_out = tf.Variable(b_alpha * tf.random_normal([MAX_CAPTCHA * CHAR_SET_LEN]))
    out = tf.add(tf.matmul(dense1_1, w_out), b_out)
    # conv1 = tf.nn.dropout(conv1, keep_prob)
    #############


    saver = tf.train.Saver()
    # predict = tf.argmax(tf.reshape(output, [-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2)
    with tf.Session() as sess:
        saver.restore(sess, "./Model/model.ckpt")
        out_res=sess.run(out, feed_dict={X: [captcha_image], keep_prob: 1})

        print 1


text, image = gen_captcha_text_and_image()
image = convert2gray(image)
image = image.flatten() / 255
predict_text = crack_captcha(image)
# print("正确: {}  预测: {}".format(text, predict_text))