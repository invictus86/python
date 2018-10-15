import pandas as pd
import numpy as np
import glob
import random
import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

tf.app.flags.DEFINE_integer('batch_size', 100, '每批次的样本数量')
tf.app.flags.DEFINE_integer('capacity', 200, '批处理队列大小')
FLAGS = tf.app.flags.FLAGS

csv_data = None

def parse_csv():
    """
    解析CSV文件, 建立文件名和标签值对应表格
    :return: None
    """
    global csv_data
    # 读取 csv标签文件
    csv_data = pd.read_csv('./data/GenPics/labels.csv', names=['index', 'chars'], index_col='index')
    # print(csv_data)

    # 增加lables列
    csv_data['labels'] = None

    # 把字母转换成标签值 A -> 0, B -> 1, ... Z -> 25
    for i, row in csv_data.iterrows():
        labels = []
        # 每一个字母转换为数字标签
        for char in row['chars']:
            # 每个字母的ascii 值 和 'A' 相减
            labels.append(ord(char) - ord('A'))

        # 把标签值添加到 表格中
        csv_data.loc[i, 'labels'] = labels

    return None

def pic_read(files):
    """
    文件队列读取图片
    :return: 图片和文件名
    """
    # 创建文件名队列
    filename_queue = tf.train.string_input_producer(files)

    # 创建读取器, 读取图片
    filename, value = tf.WholeFileReader().read(filename_queue)

    # 图片解码
    image = tf.image.decode_jpeg(value)
    print('image:', image)

    # 设置形状
    image.set_shape([20, 80, 3])

    # 批处理对列
    image_batch, filename_batch = tf.train.batch([image, filename],
            batch_size=FLAGS.batch_size, num_threads=2, capacity=FLAGS.capacity)

    return image_batch, filename_batch

def weight_var(shape, name=None):
    return tf.Variable(tf.truncated_normal(shape, mean=0.0, stddev=0.01, dtype=tf.float32), name=name)


def bias_var(shape, name=None):
    return tf.Variable(tf.zeros(shape, dtype=tf.float32), name=name)


def create_cnn_model():
    """
    创建卷积神经网络模型, 两个大的卷积层和一个全连接层
    :return: x, y_true, logits
    """
    # 定义数据占位符
    with tf.variable_scope('data'):
        x = tf.placeholder(tf.float32, [None, 20, 80, 3])
        y_true = tf.placeholder(tf.float32, [None, 4*26])

    # 卷积大层1: 卷积层, 激活函数, 池化层
    with tf.variable_scope('conv1'):
        # 卷积层: 输入: [None, 20, 80, 3]
        # 过滤器: size=[3,3], in_channel: 3, out_channels: 32, strides=1*1, padding='SAME'
        # 权重变量形状: [3, 3, 3, 32]
        # 输出的形状: [None, 20, 80, 32]
        w_conv1 = weight_var([3,3,3,32], name='w_conv1')
        b_conv1 = bias_var([32], name='b_conv1')

        x_conv1 = tf.nn.conv2d(x, filter=w_conv1, strides=[1, 1, 1, 1],
                               padding='SAME', name= 'conv1_2d') + b_conv1

        # 激活函数
        x_relu1 = tf.nn.relu(x_conv1, name='relu1')

        # 池化层: 输入形状 [None, 20, 80, 32]
        # kszie=[1, 2, 2, 1], stride =[1, 2, 2, 1]
        # 输出形状 [None, 10, 40 ,32]
        x_pool1 = tf.nn.max_pool(x_relu1, ksize=[1,2,2,1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')

    # 卷积大层2: 卷积层, 激活函数, 池化层
    with tf.variable_scope('conv2'):
        # 卷积层: 输入: [None, 10, 40, 32]
        # 过滤器: size=[3,3], in_channel: 32, out_channels: 64, strides=1*1, padding='SAME'
        # 权重变量形状: [3, 3, 32, 64]
        # 输出的形状: [None, 10, 40, 64]
        w_conv2 = weight_var([3, 3, 32, 64], name='w_conv2')
        b_conv2 = bias_var([64], name='b_conv2')

        x_conv2 = tf.nn.conv2d(x_pool1, filter=w_conv2, strides=[1, 1, 1, 1],
                               padding='SAME', name='conv2_2d') + b_conv2

        # 激活函数
        x_relu2 = tf.nn.relu(x_conv2, name='relu1')

        # 池化层: 输入形状 [None, 10, 40, 64]
        # kszie=[1, 2, 2, 1], stride =[1, 2, 2, 1]
        # 输出形状 [None, 5, 20 ,64]
        x_pool2 = tf.nn.max_pool(x_relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')


    # 全连接
    with tf.variable_scope('fc'):
        # 输入形状: [None, 5, 20, 64] => [None, 5*20*64]
        # 输出形状: [None, 4*26]
        # 权重矩阵: [5*20*64, 4*26]
        w_fc = weight_var([5*20*64, 4*26], name='w_fc')
        b_fc = bias_var([4*26])

        # 计算加权
        logits = tf.matmul(tf.reshape(x_pool2, [-1, 5*20*64]), w_fc) + b_fc

    return x, y_true, logits

def filenames_2_labels(filenames):
    """
    文件名转换为 标签值
    :param filenames:
    :return: 标签值
    """
    # 获得文件名
    labels = []
    for file in filenames:
        index, _ = os.path.splitext(os.path.basename(file))
        # 根据文件名查找标签值, 添加到 标签值列表
        labels.append(csv_data.loc[int(index), 'labels'])

    return np.array(labels)

def captcha():
    """
    卷积神经网络实现验证码识别
    :return:
    """
    # 准备文件名列表
    files = glob.glob('./data/GenPics/*.jpg')
    random.shuffle(files)

    # 分割为训练和测试图片文件列表
    # 75%作为训练图片, 25%作为测试图片
    train_num = int(len(files) * 0.75)
    train_files = files[:train_num]
    test_files = files[train_num:]

    # 文件读取流程 读取文件
    image_batch, filename_batch = pic_read(train_files)

    # 创建卷积神经网络
    x, y_true, logits = create_cnn_model()

    # 计算损失
    with tf.variable_scope('loss'):
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=logits))

    # 优化
    with tf.variable_scope('optimize'):
        # train_op = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)
        train_op = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)

    # 计算准确率
    with tf.variable_scope('accuracy'):
        equal_list = tf.reduce_all(tf.equal(tf.argmax(tf.reshape(logits, [-1, 4, 26]), axis=-1),
                     tf.argmax(tf.reshape(y_true, [-1, 4, 26]), axis=-1)), axis=-1)
        accuracy = tf.reduce_mean(tf.cast(equal_list, tf.float32))
    # 实例化模型保存类
    saver = tf.train.Saver()

    # 开启会话训练
    with tf.Session() as sess:
        # 初始全局变量
        sess.run(tf.global_variables_initializer())

        # 创建线程协调器, 启动线程
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess, coord)

        # 恢复模型
        if os.path.exists('./models/captcha/checkpoint'):
            saver.restore(sess, './models/captcha/captcha')

        # 迭代优化
        for i in range(2000):
            # 获取图片 和 文件名
            images, filenames = sess.run([image_batch, filename_batch])
            # 从文件名列表转换成标签数组
            labels = filenames_2_labels(filenames)
            labels_onehot = tf.reshape(tf.one_hot(labels, 26), [-1, 4*26]).eval()

            _, loss_value, acc = sess.run([train_op, loss, accuracy], feed_dict={x: images, y_true: labels_onehot})
            print('第 {} 次的 损失值 {} 和 准确率 {}'.format(i, loss_value, acc))

            # 保存模型
            if (i+1) % 500 == 0:
                saver.save(sess, './models/captcha/captcha')

        # 关闭线程
        coord.request_stop()
        coord.join(threads)

if __name__ == '__main__':
    # 建立 文件名和标签值的表格
    parse_csv()
    # print(csv_data)

    captcha()