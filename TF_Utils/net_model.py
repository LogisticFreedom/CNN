import tensorflow as tf

class Model(object):

    def train(self, y, yPred, lr = 1e-4):

        crossEntropy = -tf.reduce_sum(y * tf.log(yPred))  # 定义交叉熵为loss函数
        trainStep = tf.train.AdamOptimizer(lr).minimize(crossEntropy)  # 调用优化器优化
        return trainStep
