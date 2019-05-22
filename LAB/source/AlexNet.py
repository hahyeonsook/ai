import tensorflow as tf
import random

# load data
from tensorflow.examples.tutorials.mnist import input_data

tf.set_random_seed(777)

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


# hyper parameter
learning_rate = 0.001
batch_size = 32
training_epochs = 200000
display_step = 20

keep_prob = tf.placeholder(tf.float32)

# Network Parameters
n_input = 784
n_classes = 10
dropout = 0.8

# input placeholders
X = tf.placeholder(tf.float32, [None, n_input])
X_img = tf.reshape(X, [-1, 28, 28, 1])
Y = tf.placeholder(tf.float32, [None, n_classes])

# Store layers weight & bias
weights = {
  'W_c1': tf.Variable(tf.random_normal([3, 3, 1, 32])),
  'W_c2': tf.Variable(tf.random_normal([3, 3, 32, 64])),
  'W_c3': tf.Variable(tf.random_normal([3, 3, 64, 128])),
  'W_c4': tf.Variable(tf.random_normal([3, 3, 128, 256])),
  'W_c5': tf.Variable(tf.random_normal([3, 3, 256, 512])),
  'W_f1': tf.Variable(tf.random_normal([2*2*512, 2048])),
  'W_f2': tf.Variable(tf.random_normal([2048, 1024])),
  'out': tf.Variable(tf.random_normal([1024, n_classes]))
}
biases = {
  'B_c1': tf.Variable(tf.random_normal([32])),
  'B_c2': tf.Variable(tf.random_normal([64])),
  'B_c3': tf.Variable(tf.random_normal([128])),
  'B_c4': tf.Variable(tf.random_normal([256])),
  'B_c5': tf.Variable(tf.random_normal([512])),
  'B_f1': tf.Variable(tf.random_normal([2048])),
  'B_f2': tf.Variable(tf.random_normal([1024])),
  'out': tf.Variable(tf.random_normal([n_classes]))
}

# layer function
def conv2d(c_input, w, b):
  return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(c_input, w, strides=[1, 1, 1, 1], padding='SAME'), b))

def max_pool(m_input, k):
  return tf.nn.max_pool(m_input, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

def norm(n_input, size=4):
  return tf.nn.lrn(n_input, size, bias=1.0, alpha=0.001/9.0, beta=0.75)

def fully_con(f_input, w, b):
  return tf.nn.relu(tf.matmul(f_input, w) + b) //[-1, 4*4*512], [4*4*512, 2048] 

def alex_net(x, w, b, drop):
  # conv layer 1
  # Input img = [?, 28, 28, 1]
  L_c1 = conv2d(x, w['W_c1'], b['B_c1']) //64
  # max pooling
  L_c1 = max_pool(L_c1, k=2)
  # LRN
  L_c1 = norm(L_c1, size=4)
  # dropout
  L_c1 = tf.nn.dropout(L_c1, drop)

  # conv layer 2
  # Input_img = [?, 14, 14, 32]
  L_c2 = conv2d(L_c1, w['W_c2'], b['B_c2'])
  # max pooling
  L_c2 = max_pool(L_c2, k=2)
  # LRN
  L_c2 = norm(L_c2, size=4)
  # dropout
  L_c2 = tf.nn.dropout(L_c2, drop)

  # conv layer 3
  # Input Img = [?, 7, 7, 64]
  L_c3 = conv2d(L_c2, w['W_c3'], b['B_c3'])
  # dropout
  L_c3 = tf.nn.dropout(L_c3, drop)

  # conv layer 4
  # Input_Img = [?, 4, 4, 128]
  L_c4 = conv2d(L_c3, w['W_c4'], b['B_c4'])
  # dropout
  L_c4 = tf.nn.dropout(L_c4, drop)

  # conv layer 5
  # Input_Img = [?, 2, 2, 256]
  L_c5 = conv2d(L_c4, w['W_c5'], b['B_c5'])
  # max pooling
  L_c5 = max_pool(L_c5, k=2)
  # dropout
  L_c5 = tf.nn.dropout(L_c5, drop)

  # flat
  L_flat = tf.reshape(L_c5, [-1, w['W_f1'].get_shape().as_list()[0]])

  # FC 1
  L_f1 = fully_con(L_flat, w['W_f1'], b['B_f1']) // 4*4*512, 4*4*512/2048, 2048

  # FC 2
  L_f2 = fully_con(L_f1, w['W_f2'], b['B_f2'])

  # FC 3
  return fully_con(L_f2, w['W_f1'], b['B_f1'])


logits = alex_net(X, weights, biases, keep_prob)

# cost
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
# optimizer
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(logits,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# train
print('AlexNet Learning started. It takes sometime.')
with tf.Session() as sess:
  # Initialize
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
