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

# conv factory 1
# input Img = [?, 28, 28, 1]
W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01))
# conv layer
L1 = tf.nn.conv2d(X_img, W1, strides=[1, 1, 1, 1], padding = 'SAME')
# activation function
L1 = tf.nn.relu(L1)
# pooling
L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1],
                    strides=[1, 2, 2, 1], padding = 'SAME')
# dropout
L1 = tf.nn.dropout(L1, keep_prob=keep_prob)

# conv factory 2
# input Img = [?, 14, 14, 32]
W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
# conv layer
L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding = 'SAME')
# activation function
L2 = tf.nn.relu(L2)
# pooling
L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1],
                    strides=[1, 2, 2, 1], padding = 'SAME')
# dropout
L2 = tf.nn.dropout(L2, keep_prob=keep_prob)

# conv factory 3
# input Img = [?, 7, 7, 64]
W3 = tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=0.01))
# conv layer
L3 = tf.nn.conv2d(L2, W3, strides=[1, 1, 1, 1], padding = 'SAME')
# activation function
L3 = tf.nn.relu(L3)
# pooling
L3 = tf.nn.max_pool(L3, ksize=[1, 2, 2, 1],
                    strides=[1, 2, 2, 1], padding = 'SAME')
# dropout
L3 = tf.nn.dropout(L3, keep_prob=keep_prob)
# flat
L3_flat = tf.reshape(L3, [-1, 128*4*4])

# FC 1
W4 = tf.get_variable("W4", shape=[128*4*4, 625],
                     #  초기화 알고리즘, xavier initializer
                     initializer = tf.contrib.layers.xavier_initializer())
# Variables
b4 = tf.Variable(tf.random_normal([625]))
# hypersis
L4 = tf.nn.relu(tf.matmul(L3_flat, W4) + b4)
# dropout
L4 = tf.nn.dropout(L4, keep_prob=keep_prob)

# FC 2
# Variables
W5 = tf.get_variable("W5", shape=[625, 10],
                     initializer = tf.contrib.layers.xavier_initializer())
# hypersis
b5 = tf.Variable(tf.random_normal([10]))
# logits
logits = tf.matmul(L4, W5) + b5

# cost
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = Y))
# optimizer
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)

# initialize
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# train
print('Convolution Neural Networks Learning started. It takes sometime.')
for epoch in range(training_epochs):
  # initialize
  avg_cost = 0
  total_batch = int(mnist.train.num_examples / batch_size)

  for i in range(total_batch):
    batch_xs, batch_ys = mnist.train.next_batch(batch_size)
    feed_dict = {X: batch_xs, Y: batch_ys, keep_prob: 0.7}
    c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
    avg_cost += c / total_batch

  print('Epoch:', '%04d' % (epoch + 1), 'cost=', '{:.9f}'.format(avg_cost))

print('Learning Finished!')
