import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# 定义0阶张量
a = tf.constant(3.0, tf.float32)

# 定义1阶张量
b = tf.constant([3.0, 4.0], tf.float32)

# 定义2阶张量
c = tf.constant([[3,4], [5,6]], tf.float32)

# print('a:', a)
# print('b:', b)
# print('c:', c)
#
# print('c.op:\n', c.op)

# 张量的静态形状
t1 = tf.placeholder(tf.float32, shape=[10,10])
t2 = tf.placeholder(tf.float32, shape=[None, 10])
t3 = tf.placeholder(tf.float32, shape=[None, None])

# print('t1:', t1)
# print('t2:', t2)
# print('t3:', t3)

# 设置张量的静态形状的不确定部分
# t1.set_shape([2,50])
t2.set_shape([2, 10])
t3.set_shape([2, 2])

print('t1:', t1)
print('t2:', t2)
print('t3:', t3)

# 动态形状
# t4 = tf.reshape(t1, [2, 50])
t5 = tf.reshape(t1, [100])
print('t5:', t5)
