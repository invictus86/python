import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


# tensorflow 加法操作
a = tf.constant(3.0)
b = tf.constant(4.0)

# print('a:', a)
# print('b:', b)

c = tf.add(a, b)
# print('c:', c)

# 获得结果
with tf.Session() as sess:
    print('结果:', sess.run(c))
    print('会话指向的图:', sess.graph)

# 获取默认的图
print('默认的图:', tf.get_default_graph())
print('张量数据属于的图', a.graph)

# 创建新的图
g = tf.Graph()

# 添加张量
with g.as_default():
    d = tf.constant(10.0)
    f = tf.constant(11.0)
    s = tf.add(d,f)
    print('s的图:', s.graph)

with tf.Session() as sess:
    sess.run(s)