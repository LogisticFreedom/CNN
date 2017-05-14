import tensorflow as tf
from TF_Utils.util import conv2d, maxPool, weightVariable, biasVariable, convLayer, denseLayer
from TF_Utils.net_model import Model

class AlexNetModel(Model):

    def __init__(self):
        pass

    def alexnetModel(self, x, train = True):

        # 数据格式转化
        x = tf.reshape(x, [-1, 227, 227, 3])

        # 第一层卷积+LRN+池化
        h_conv1 = convLayer(x, 11, 11, 4, 4, 3, 96, pad="VALID", name="conv1")
        norm_1 = tf.nn.local_response_normalization(h_conv1, name="norm1")
        h_pool1 = maxPool(norm_1, 3, 2, pad="VALID", name="pool1")

        # 第二层卷积+LRN+池化
        h_conv2 = convLayer(h_pool1, 5, 5, 1, 1, 96, 256, pad="SAME", name="conv2")
        norm_2 = tf.nn.local_response_normalization(h_conv2, name="norm2")
        h_pool2 = maxPool(norm_2, 3, 2, pad="VALID", name="pool2")

        # 连续三层卷积
        h_conv3 = convLayer(h_pool2, 3, 3, 1, 1, 256, 384, pad="SAME", name="conv3")
        h_conv4 = convLayer(h_conv3, 3, 3, 1, 1, 384, 384, pad="SAME", name="conv4")
        h_conv5 = convLayer(h_conv4, 3, 3, 1, 1, 384, 256, pad="SAME", name="conv5")

        # 池化
        h_pool5 = maxPool(h_conv5, 3, 2, pad="VALID", name="pool3")

        # 展平
        length = h_pool5.get_shape()[1] * h_pool5.get_shape()[2] * h_pool5.get_shape()[3]
        print(length)
        fc_h0 = tf.reshape(h_pool5, [-1, 256])  # 能不能动态的展开?

        # 第一层全连接
        fc_h1 = denseLayer(fc_h0, 256, 4096, name="fc1")

        # 第二层全连接+dropout
        if train:
            fc_h1 = tf.nn.dropout(fc_h1, 0.8)
        fc_h2 = denseLayer(fc_h1, 4096, 4096, name="fc2")

        # 第三层全连接softmax输出+dropout
        if train:
            fc_h2 = tf.nn.dropout(fc_h2, 0.8)
        output = denseLayer(fc_h2, 4096, 1000, active="softmax", name="softmax")

        return output





