import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt

# 自定义命令行参数
tf.app.flags.DEFINE_integer('steps', 200, '迭代的次数')
tf.app.flags.DEFINE_string('model_path', './models/lr/lrmodel', '模型存放路径')

FLAGS = tf.app.flags.FLAGS

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# 人造数据
# 特征值 100个在[-1, 1], 均匀分布
x = np.random.uniform(-1, 1, 100)
# 目标值, y= 0.7 * x + 0.5 + 正态分布
y = x*0.7 + 0.5 + np.random.normal(0.0, 0.1, 100)

# plt.figure()
# plt.scatter(x, y)
# plt.show()

def linear_regression(argv):
    """
    线性回归方法确定权重和偏执
    :return: None
    """

    # 定义数据流图
    # 定义数据张量, 占位符
    with tf.variable_scope('data'):
        tx = tf.placeholder(tf.float32, [None, 1], name='tensor_x')
        ty = tf.placeholder(tf.float32, [None, 1], name='tensor_y')

    # 线性模型 预测
    # [None, 1] * [1, 1] + b = y_pred
    # 定义权重和偏执
    with tf.variable_scope('model'):
        weight = tf.Variable(tf.truncated_normal([1,1], dtype=tf.float32, mean=0.0, stddev=0.1), name='weight')
        bias = tf.Variable(tf.zeros([1], dtype=tf.float32), name='bias')

        y_pred = tf.matmul(tx, weight) + bias
        # print(y_pred)

    # 计算损失值
    with tf.variable_scope('loss'):
        loss = tf.reduce_mean(tf.square(ty - y_pred))

    # 优化
    with tf.variable_scope('optimize'):
        train_op = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)
    # print(train_op)

    # 收集 变量
    tf.summary.scalar('losses', loss)
    tf.summary.histogram('weights', weight)
    tf.summary.histogram('biases', bias)

    # 合并
    merged = tf.summary.merge_all()
    print(merged)

    # 实例化保存文件类
    saver = tf.train.Saver()

    # 开启会话训练
    with tf.Session() as sess:
        # 初始化全局变量
        sess.run(tf.global_variables_initializer())

        file_writer = tf.summary.FileWriter('./logs/lr/', graph=sess.graph)
        print('初始化的权重 {} 和偏置 {}:'.format(weight.eval(), bias.eval()))

        # 加载模型
        if os.path.exists('./models/lr/checkpoint'):
            saver.restore(sess, FLAGS.model_path)
            print('加载之后的权重 {} 和 偏置{}'.format(weight.eval(), bias.eval()))

        # 迭代更新 权重和偏执, 优化, 计算损失
        for i in range(FLAGS.steps):
            _, loss_value, summary = sess.run([train_op, loss, merged], feed_dict={tx: x.reshape(-1, 1), ty: y.reshape(-1, 1)})
            print('运行第 {} 次后的权重{} 和偏置 {}, 损失值 {}'.format(i, weight.eval(), bias.eval(), loss_value))
            file_writer.add_summary(summary, i)

            # 每50次保存一次模型
            if (i+1)%50 ==0:
                saver.save(sess, FLAGS.model_path)

    return None

if __name__ == '__main__':
    # linear_regression()
    tf.app.run(linear_regression)