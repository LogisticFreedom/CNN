import tensorflow as tf
import numpy as np
from TF_Utils.util import conv2d, maxPool, weightVariable, biasVariable, convLayer, denseLayer
from TF_Utils.net_model import Model

class LeNetModel(Model):

    def __init__(self):
        pass

    def LeNetmodel(self, x, train = True):

        # 变换数据
        x = tf.reshape(x, [-1, 28, 28, 1])

        # 第一层卷积加池化
        h_conv1 = convLayer(x, 5, 5, 1, 1, 1, 6, pad="VALID", name="conv1")
        h_pool1 = maxPool(h_conv1, name="pool1")

        # 第二次卷积加池化
        h_conv2 = convLayer(h_pool1, 5, 5, 1, 1, 6, 16, pad="VALID", name="conv2")
        h_pool2 = maxPool(h_conv2, name="pool2")

        # 展平
        length = h_pool2.get_shape()[1] * h_pool2.get_shape()[2]*h_pool2.get_shape()[3]
        print(length)
        fc_h0 = tf.reshape(h_pool2, [-1, 256]) # 能不能动态的展开?

        # 第一层全连接
        fc_h1 = denseLayer(fc_h0, 256, 120, name="fc1")

        # 第二层全连接
        fc_h2 = denseLayer(fc_h1, 120, 84, name="fc2")

        if train:
            fc_h2 = tf.nn.dropout(fc_h2, 0.5)

        # 第三层全连接softmax输出
        output = denseLayer(fc_h2, 84, 10, active="softmax", name="softmax")

        return output

    # def train(self, y, yPred, lr = 1e-4):
    #
    #     crossEntropy = -tf.reduce_sum(y * tf.log(yPred))  # 定义交叉熵为loss函数
    #     trainStep = tf.train.AdamOptimizer(lr).minimize(crossEntropy)  # 调用优化器优化
    #     return trainStep







