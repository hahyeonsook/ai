'''
Placeholder: 연산 노드를 가리키는 텐서, 그래프를 실행할 때 사용자가 데이터를 주입할 수 있는 통로
Variable: 변수, tensorflow 연산에 의해 채워지는 텐서

https://bcho.tistory.com/1154
'''
import tensorflow as tf
import random

# load data
from tensorflow.examples.tutorials.mnist import input_data

tf.set_random_seed(777)

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# hyper parameters
learning_rate = 0.001
batch_size = 100
training_epochs = 15
# dropout을 할 때, 연결 비율
keep_prob = tf.placeholder(tf.float32)

# input place holders
X = tf.placeholder(tf.float32, [None, 784])
X_img = tf.reshape(X, [-1, 28, 28, 1])  # img 28*28*1(black/white)
Y = tf.placeholder(tf.float32, [None, 10])

# conv factory 1
# input Img shape = [?, 28, 28, 1]
# 3*3 크기의 색깔 값을 1을 가진 32 개의 filter로 
W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01))

# conv layer
L1 = tf.nn.conv2d(X_img, W1, strides=[1, 1, 1, 1], padding = 'SAME')

# activation function
L1 = tf.nn.relu(L1)

# pooling layer
L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding = 'SAME')

# dropout, overfitting 방지
L1 = tf.nn.dropout(L1, keep_prob=keep_prob)

# conv factory 2
# input Img shape = [?, 14, 14, 32]
# 3*3 크기의 32개의 값을 가진 64개의 filter로
W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))

# conv layer
L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding = 'SAME')

# activation function
L2 = tf.nn.relu(L2)

# pooling layer
L2 = tf.nn.max_pool(L2, ksize = [1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding = 'SAME')

# dropout
L2 = tf.nn.dropout(L2, keep_prob=keep_prob)

# conv factory 3
# input Img = [?, 7, 7, 64]
# 3*3 크기의 64개의 값을 가진 128개의 filter로
W3 = tf.Variable(tf.random_normal([3, 3, 64, 128], stddev = 0.01))

# conv layer
L3 = tf.nn.conv2d(L2, W3, strides=[1, 1, 1, 1], padding = 'SAME')

# activation function
L3 = tf.nn.relu(L3)

# pooling layer
L3 = tf.nn.max_pool(L3, ksize = [1, 2, 2, 1],
                        strides = [1, 2, 2, 1], padding = 'SAME')
L3 = tf.nn.dropout(L3, keep_prob=keep_prob)

# flat
L3_flat = tf.reshape(L3, [-1, 128 * 4 * 4])


# fully connected layer 1
# name filed 값과 동일한 텐서가 존재할 경우, 새로 만들지 않고 기존 텐서를 불러들임
W4 = tf.get_variable("W4", shape=[128*4*4, 625],
                     initializer = tf.contrib.layers.xavier_initializer())
b4 = tf.Variable(tf.random_normal([625]))
L4 = tf.nn.relu(tf.matmul(L3_flat, W4) + b4)
L4 = tf.nn.dropout(L4, keep_prob=keep_prob)

# fully connected layer 2
W5 = tf.get_variable("W5", shape=[625, 10],
                     initializer = tf.contrib.layers.xavier_initializer())
b5 = tf.Variable(tf.random_normal([10]))
logits = tf.matmul(L4, W5) + b5

# define cost/loss & optimizer
# cross entropy를 logit으로 softmax해서 결과를 냄
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)

# initialize
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# train
print('Learning started. It takes sometime.')

for epoch in range(training_epochs):
  avg_cost = 0
  total_batch = int(mnist.train.num_examples / batch_size)

  for i in range(total_batch):
    batch_xs, batch_ys = mnist.train.next_batch(batch_size)
    feed_dict = {X: batch_xs, Y: batch_ys, keep_prob: 0.7}
    c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
    avg_cost += c / total_batch

  print('Epoch:', '%04d' % (epoch+1), 'cost =', '{:.9f}'.format(avg_cost))

print('Learning Finished!')



