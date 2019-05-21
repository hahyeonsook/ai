import tensorflow as tf
import random

# load data
from tensorflow.examples.tutorials.mnist import input_data

tf.set_random_seed(777)

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

''' gpu memory
config=tf.ConfigProto()
config.gpu_options.allocator_type='BFC'
config.gpu_options.per_process_gpu_memory_fraction = 0.90
'''

# hyper parameter
learning_rate = 0.001
batch_size = 64
training_epochs = 200000
display_step = 20

# Network Parameters
dropout = 0.8

keep_prob = tf.placeholder(tf.float32)

# input placeholders
X = tf.placeholder(tf.float32, [None, 784])
X_img = tf.reshape(X, [-1, 28, 28, 1])
Y = tf.placeholder(tf.float32, [None, 10])

# AlexNet Model
def conv2d(c_input, w, b):
  return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(c_input, w, strides=[1, 1, 1, 1], padding='SAME'), b))

def max_pool(m_input, k):
  return tf.nn.max_pool(m_input, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

def norm(n_input, size=4):
  return tf.nn.lrn(n_input, size, bias=1.0, alpha=0.001/9.0, beta=0.75)

def alex_net(x, drop):
  # conv layer
  W_c1 = tf.Variable(tf.random_normal([3, 3, 1, 64], stddev = 0.01))
  b_c1 = tf.Variable(tf.random_normal([64]))

  L_c1 = conv2d(X_img, W_c1, b_c1)
  # max pooling
  L_c1 = max_pool(L_c1, k=2)
  # LRN
  L_c1 = norm(L_c1, size=4)
  # dropout
  L_c1 = tf.nn.dropout(L_c1, drop)

  # conv layer
  W_c2 = tf.Variable(tf.random_normal([3, 3, 64, 128], stddev = 0.01))
  b_c2 = tf.Variable(tf.random_normal([128]))

  L_c2 = conv2d(L_c1, W_c2, b_c2)
  # max pooling
  L_c2 = max_pool(L_c2, k=2)
  # LRN
  L_c2 = norm(L_c2, size=4)
  # dropout
  L_c2 = tf.nn.dropout(L_c2, drop)

  # conv layer
  W_c3 = tf.Variable(tf.random_normal([3, 3, 128, 256], stddev = 0.01))
  b_c3 = tf.Variable(tf.random_normal([256]))

  L_c3 = conv2d(L_c2, W_c3, b_c3)
  # dropout
  L_c3 = tf.nn.dropout(L_c3, drop)

  # conv layer
  W_c4 = tf.Variable(tf.random_normal([3, 3, 256, 512], stddev = 0.01))
  b_c4 = tf.Variable(tf.random_normal([512]))

  L_c4 = conv2d(L_c3, W_c4, b_c4)
  # dropout
  L_c4 = tf.nn.dropout(L_c4, drop)

  # conv layer
  W_c5 = tf.Variable(tf.random_normal([3, 3, 512, 1024], stddev = 0.01))
  b_c5 = tf.Variable(tf.random_normal([1024]))

  L_c5 = conv2d(L_c4, W_c5, b_c5)
  # max pooling
  L_c5 = max_pool(L_c5, k=2)
  # dropout
  L_c5 = tf.nn.dropout(L_c5, drop)

  # flat
  L_flat = tf.reshape(L_c5, [-1, 1024*4*4])

  # FC
  W_f1 = tf.Variable(tf.random_normal([1024*4*4, 2048], stddev = 0.01))
  b_f1 = tf.Variable(tf.random_normal([2048]))

  L_f1 = tf.nn.relu(tf.matmul(L_flat, W_f1) + b_f1)

  # FC
  W_f2 = tf.Variable(tf.random_normal([2048, 1024], stddev = 0.01))
  b_f2 = tf.Variable(tf.random_normal([1024]))

  L_f2 = tf.nn.relu(tf.matmul(L_f1, W_f2) + b_f2)

  # FC
  W_f3 = tf.Variable(tf.random_normal([1024, 10], stddev = 0.01))
  b_f3 = tf.Variable(tf.random_normal([10]))

  return tf.nn.relu(tf.matmul(L_f2, W_f3) + b_f3)


logits = alex_net(X, keep_prob)

# cost
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
# optimizer
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)

# initialize
sess = tf.Session()
sess.run(tf.global_variables_initializer())
#step = 1

# train
print('AlexNet Learning started. It takes sometime.')
with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  step = 1
  while step * batch_size < training_epochs:
    batch_xs, batch_ys = mnist.train.next_batch(batch_size)
    sess.run(optimizer, feed_dict={X: batch_xs, Y: batch_ys, keep_prob: 1.})
    if step % display_step == 0:
      # Calculate batch accuracy
      acc = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
      # Calculate batch loss
      loss = sess.run(cost, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
      print('epoch: '+str(step*batch_size) + ', Minibatch loss= ' + "{:.6f}".format(loss) + ", Training Accuracy= " + "{:.5f}".format(acc))
    step += 1

  print('Learning Finished!')
  print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: mnist.test.images[:256], y: mnist.test.labels[:256], keep_prob: 1.}))
