import tensorflow as tf

def weightVariable(shape, std=0.1, mean=0, name=None):

    initial = tf.truncated_normal(shape,stddev=std, mean=mean, name=name) #截断正态分布，此函数原型为尺寸、均值、标准差
    return tf.Variable(initial)

def biasVariable(shape,  value=0.1, name=None):
    initial = tf.constant(value,shape=shape, name=name)
    return tf.Variable(initial)

def conv2d(x, W, xstep = 1, ystep=1, pad="SAME", name=None):
    return tf.nn.conv2d(x,W,strides=[1, xstep, ystep, 1],padding=pad, name=name) # strides第0位和第3为一定为1，剩下的是卷积的横向和纵向步长

def maxPool(x, ksize = 2 , step = 2, pad="SAME", name=None):
    return tf.nn.max_pool(x,ksize = [1, ksize, ksize, 1],strides=[1, step, step, 1],padding=pad, name=name)# 参数同上，ksize是池化块的大小

def convLayer(x, wH, wW, xstep, ystep, channelNum, kernelNum, pad="SAME", active = "relu", name=None):

    with tf.name_scope(name):
        W = weightVariable([wH, wW, channelNum, kernelNum])
        b = biasVariable([kernelNum])
        convOutput = conv2d(x, W, xstep=xstep, ystep=ystep, pad=pad)+b
        if active == "relu":
            return tf.nn.relu(convOutput)
        if active == "sigmoid":
            return tf.nn.sigmoid(convOutput)
        if active == "tanh":
            return tf.nn.tanh(convOutput)

def denseLayer(x, inputDim, outputDim, active="relu", name=None):

    w = weightVariable([inputDim, outputDim])
    b = biasVariable([outputDim])
    output = tf.nn.relu(tf.matmul(x, w) + b, name=name)
    if active == "relu":
        return tf.nn.relu(output)
    if active == "sigmoid":
        return tf.nn.sigmoid(output)
    if active == "tanh":
        return tf.nn.tanh(output)
    if active == "softmax":
        return tf.nn.softmax(output)
