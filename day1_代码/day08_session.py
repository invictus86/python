import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
#
a = tf.constant(3.0, name='const_a')
b = tf.constant(4.0, name='const_b')
c = tf.add(a,b, name='sum_c')
#
# d = tf.constant(10.0)
# f = tf.constant(11.0)
# g  = tf.multiply(d,f)
# # 开启会话, 运行, 最后需要关闭会话释放资源
# # sess = tf.Session()
# # print(sess.run(c))
# # sess.close()
#
# # 上下文管理器
# with tf.Session() as sess:
#     print(sess.run([c, g]))

# 定义占位符
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)

s = tf.add(input1, input2)

with tf.Session() as sess:
    s_value = sess.run(s, feed_dict={input1: 10.0, input2: 11.0})
    print(s_value)

print('a的名称:', a.name)
print('b的名称:', b.name)
print('c的名称:', c.name)