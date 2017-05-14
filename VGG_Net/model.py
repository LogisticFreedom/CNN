import tensorflow as tf
from TF_Utils.util import conv2d, maxPool, weightVariable, biasVariable, convLayer, denseLayer
from TF_Utils.net_model import Model

class VGGNetModel(Model):

    def __init__(self):
        pass

    def vggnetModel_19(self, x, train=True):

        # 数据格式转化
        x = tf.reshape(x, [-1, 224, 224, 3])

        # 第一组卷积+池化
        with tf.name_scope("conv_group1"):
            h_conv1 = convLayer(x, 3, 3, 1, 1, 3, 64, pad="SAME", name="conv1")
            h_conv2 = convLayer(h_conv1, 3, 3, 1, 1, 3, 64, pad="SAME", name="conv2")
            h_pool1 = maxPool(h_conv2, 2, 2, pad="VALID", name="pool1")

        # 第二组卷积+池化
        with tf.name_scope("conv_group2"):
            h_conv3 = convLayer(h_pool1, 3, 3, 1, 1, 64, 128, pad="SAME", name="conv3")
            h_conv4 = convLayer(h_conv3, 3, 3, 1, 1, 64, 128, pad="SAME", name="conv4")
            h_pool2 = maxPool(h_conv4, 2, 2, pad="VALID", name="pool2")

        # 第二组卷积+池化
        with tf.name_scope("conv_group3"):
            h_conv5 = convLayer(h_pool2, 3, 3, 1, 1, 128, 256, pad="SAME", name="conv5")
            h_conv6 = convLayer(h_conv5, 3, 3, 1, 1, 128, 256, pad="SAME", name="conv6")
            h_conv7 = convLayer(h_conv6, 3, 3, 1, 1, 128, 256, pad="SAME", name="conv7")
            h_conv8 = convLayer(h_conv7, 3, 3, 1, 1, 128, 256, pad="SAME", name="conv8")
            h_pool3 = maxPool(h_conv8, 2, 2, pad="VALID", name="pool2")

        # 第四组卷积+池化
        with tf.name_scope("conv_group4"):
            h_conv9 = convLayer(h_pool3, 3, 3, 1, 1, 256, 512, pad="SAME", name="conv9")
            h_conv10 = convLayer(h_conv9, 3, 3, 1, 1, 256, 512, pad="SAME", name="conv10")
            h_conv11 = convLayer(h_conv10, 3, 3, 1, 1, 256, 512, pad="SAME", name="conv11")
            h_conv12 = convLayer(h_conv11, 3, 3, 1, 1, 256, 512, pad="SAME", name="conv12")
            h_pool4 = maxPool(h_conv12, 2, 2, pad="VALID", name="pool4")

        # 第五组卷积+池化
        with tf.name_scope("conv_group5"):
            h_conv13 = convLayer(h_pool4, 3, 3, 1, 1, 512, 512, pad="SAME", name="conv13")
            h_conv14 = convLayer(h_conv13, 3, 3, 1, 1, 512, 512, pad="SAME", name="conv14")
            h_conv15 = convLayer(h_conv14, 3, 3, 1, 1, 512, 512, pad="SAME", name="conv15")
            h_conv16 = convLayer(h_conv15, 3, 3, 1, 1, 512, 512, pad="SAME", name="conv16")
            h_pool5 = maxPool(h_conv16, 2, 2, pad="VALID", name="pool5")

        # 展平
        length = h_pool5.get_shape()[1] * h_pool5.get_shape()[2] * h_pool5.get_shape()[3]
        print(length)
        fc_h0 = tf.reshape(h_pool5, [-1, 256])  # 能不能动态的展开?

        # 第一层全连接
        fc_h1 = denseLayer(fc_h0, 256, 4096, name="fc1")

        # 第二层全连接+dropout
        if train:
            fc_h1 = tf.nn.dropout(fc_h1, 0.5)
        fc_h2 = denseLayer(fc_h1, 4096, 4096, name="fc2")

        # 第三层全连接softmax输出+dropout
        if train:
            fc_h2 = tf.nn.dropout(fc_h2, 0.5)
        output = denseLayer(fc_h2, 4096, 1000, active="softmax", name="softmax")

        return output