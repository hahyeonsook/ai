import tensorflow as tf
import random

# load data
from tensorflow.examples.tutorials.mnist import input_data

tf.set_random_seed(777)

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# hyper parameter
learning_rate = 0.001
batch_size = 100
training_epochs = 15

keep_prob = tf.placeholder(tf.float32)

# input placeholders
X = tf.placeholder(tf.float32, [None, 784])
X_img = tf.reshape(X, [-1, 28, 28, 1])
Y = tf.placeholder(tf.float32, [None, 10])


