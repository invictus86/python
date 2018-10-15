import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

a = tf.constant(3.0, name='const_a')
b = tf.constant(4.0, name='const_b')
c = tf.add(a,b, name='sum_c')

d = tf.constant(10.0, name='const_d')
e = tf.constant(11.0, name='const_e')
f = tf.multiply(d,e)

g = tf.subtract(c,f)


with tf.Session() as sess:
    g_value = sess.run(g)

    # 添加tensorboard时间文件
    tf.summary.FileWriter('./logs/tb', graph=sess.graph)
    
    print(g_value)