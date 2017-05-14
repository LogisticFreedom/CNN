import tensorflow as tf
from LeNet_5.model import LeNetModel

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

x = tf.placeholder(tf.float32, shape=[None, 784])
y = tf.placeholder(tf.float32, shape=[None, 10])

model = LeNetModel()

ypred = model.LeNetmodel(x)
trainStep = model.train(y, ypred)

correct_prediction = tf.equal(tf.argmax(ypred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.InteractiveSession()
init = tf.global_variables_initializer()
sess.run(init)

for i in range(10000):
    batch = mnist.train.next_batch(100)
    if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x: batch[0], y: batch[1]})
        print("step %d, training accuracy %g" % (i, train_accuracy))
    trainStep.run(feed_dict={x: batch[0], y: batch[1]})

testPred = model.LeNetmodel(x, train=False)
testcorrect_prediction = tf.equal(tf.argmax(testPred, 1), tf.argmax(y, 1))
testaccuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print("test accuracy %g" % testaccuracy.eval(feed_dict={x: mnist.test.images[0:500], y: mnist.test.labels[0:500]}))