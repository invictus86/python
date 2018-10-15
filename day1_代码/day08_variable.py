import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# 定义变量
# 常数作为初始值
var1 = tf.Variable([1,2,3,4], name='var1')
# 用随机数进行初始化
var2 = tf.Variable(tf.random_normal([2,3], dtype=tf.float32), name='var2')

# 调用assign赋值
var3 = var2.assign([[3.0, 4.0, 5.0], [1.0, 2.0, 3.0]])

# 通过 get_variable
var4 = tf.get_variable('my_var', [2,3], tf.float32)

print(var1)
print(var2)

# 定义初始化全局变量操作
# init_op = tf.global_variables_initializer()

# 运行会话
with tf.Session() as sess:
    # 初始化全局变量
    sess.run(tf.global_variables_initializer())

    var1_value, var2_value, var3_value, var4_value = sess.run([var1, var2, var3, var4])
    print(var1_value)
    print(var2_value)
    print(var3_value)
    print(var4_value)
