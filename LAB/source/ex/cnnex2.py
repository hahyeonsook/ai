import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('./MNIST_data', one_hot=True, reshape=False)

X = tf.placeholer(tf.float32, shape=[None, 28, 28, 1])
Y_Label = tf.placeholder(tf.float32, shape=[None, 10])

Kernel1 = tf.Variable(tf.truncated_normal(shape=[4, 4, 1, 4], stddev=0.1))
Bias1 = tf.Variable(tf.truncated_normal(shape=[4], stddev=0.1))
Conv1 = tf.nn.conv2d(X, Kernel1, strides=[1, 1, 1, 1], padding='SAME') + Bias1
Activation1 = tf.nn.relu(Conv1)
Pool1 = tf.nn.max_pool(Activation1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

Kernel2 = tf.Variable(tf.truncated_normal(shape=[4, 4, 1, 4], stddev=0.1))
Bias2 = tf.Variable(tf.truncated_normal(shape=[4], stddev=0.1))
Conv2 = tf.nn.conv2d(Pool1, Kernel2, strides=[1, 1, 1, 1], padding='SAME') + Bias2
Activation2 = tf.nn.relu(Conv2)
Pool2 = tf.nn.max_pool(Activation2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

W1 = tf.Variable(tf.truncated_normal(shape=[8, 7, 7, 10]))
B1 = tf.Variable(tf.truncated_normal(shape=[10]))
Pool2_flat = tf.reshape(Pool2, [-1, 8*7*7])
# batch_size 한 번의 batch마다 주는 데이터 샘플의 size. 여기서 batch는 나눠진 데이터 셋을 뜻함
# reshape, 원하는 shape를 직접 입력하여 바꿀 수 있다. shape에 -1을 입력하면 고정된 차원(8*7*7)은 우선 채우고 남은 부분은 알아서 채워준다.
# 마지막 차원이  8*7*7개의 값으로 고정된 2차원 행렬을 만듦
OutputLayer = tf.matmul(Pool2_flat, W1) + B1

Loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y_Label, logits=OutputLayer))
train_step = tf.train.AdamOptimizer(0.005).minimize(Loss)

correct_prediction = tf.equal(tf.argmax(OutputLayer, 1), tf.argmax(Y_Label, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session as sess:
  print("START")
  # Variables 초기화
  sess.run(tf.global_variables_initializer())
  for i in range(10000):
    trainingData, Y = mnist.train.next_batch(64)
    sess.run(train_step, feed_dict={X:trainingData, Y_Label:Y})
    if i%100 :
      # feed_dict={}에는 placeholder 로 정의한 변수들의 값을 직접 넣어주는 것
      print(sess.run(accuracy, feed_dict={X:mnist.test.images, Y_Label:mnist.test.labels}))


