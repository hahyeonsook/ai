# conv factory 1
# input Img=[?, 28, 28, 1]
W_c1 = tf.Variable(tf.random_normal([3, 3, 1, 64], stddev = 0.01))
b_c1 = tf.Variable(tf.random_normal([64]))

L_c1 = tf.nn.bias_add(tf.nn.conv2d(X_img, W_c1, strides=[1, 1, 1, 1], padding='SAME'), b_c1)
L_c1 = tf.nn.relu(L_c1)
# maxpooling
L_c1 = tf.nn.max_pool(L_c1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
# LRN 1
L_c1 = tf.nn.lrn(L_c1, 4, bias=1.0, alpha=0.001/9.0, beta=0.75)
# dropout
L_c1 = tf.nn.dropout(L_c1, keep_prob=keep_prob)

# Conv factory 2
# input Img=[?, 14, 14, 64]
W_c2 = tf.Variable(tf.random_normal([3, 3, 64, 128], stddev = 0.01))
b_c2 = tf.Variable(tf.random_normal([128]))

L_c2 = tf.nn.bias_add(tf.nn.conv2d(L_c1, W_c2, strides=[1, 1, 1, 1], padding='SAME'), b_c2)
L_c2 = tf.nn.relu(L_c2)
# maxpooling
L_c2 = tf.nn.max_pool(L_c2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
# LRN 2
L_c2 = tf.nn.lrn(L_c2, 4, bias=1.0, alpha=0.001/9.0, beta=0.75)
# dropout
L_c2 = tf.nn.dropout(L_c2, keep_prob=keep_prob)

# Conv factory 3
# input Img=[?, 7, 7, 128]
W_c3 = tf.Variable(tf.random_normal([3, 3, 128, 256], stddev = 0.01))
b_c3 = tf.Variable(tf.random_normal([256]))

L_c3 = tf.nn.bias_add(tf.nn.conv2d(L_c2, W_c3, strides=[1, 1, 1, 1], padding='SAME'), b_c3)
L_c3 = tf.nn.relu(L_c3)
# dropout
L_c3 = tf.nn.dropout(L_c3, keep_prob=keep_prob)

# Conv factory 4
# input Img=[?, 7, 7, 256]
W_c4 = tf.Variable(tf.random_normal([3, 3, 256, 512], stddev = 0.01))
b_c4 = tf.Variable(tf.random_normal([512]))

L_c4 = tf.nn.bias_add(tf.nn.conv2d(L_c3, W_c4, strides=[1, 1, 1, 1], padding='SAME'), b_c4)
L_c4 = tf.nn.relu(L_c4)
# dropout
L_c4 = tf.nn.dropout(L_c4, keep_prob=keep_prob)

# Conv factory 5
# input Img=[?, 7, 7, 512]
W_c5 = tf.Variable(tf.random_normal([3, 3, 512, 1024], stddev = 0.01))
b_c5 = tf.Variable(tf.random_normal([1024]))

L_c5 = tf.nn.bias_add(tf.nn.conv2d(L_c4, W_c5, strides=[1, 1, 1, 1], padding='SAME'), b_c5)
L_c5 = tf.nn.relu(L_c5)
# dropout
L_c5 = tf.nn.dropout(L_c5, keep_prob=keep_prob)
# maxpooling
L_c5 = tf.nn.max_pool(L_c5, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
#flat
# input Img=[?, 4, 4, 1024] -> [-1, 4*4*1024]
L_flat = tf.reshape(L_c5, [-1, 1024*4*4])

# FC 1
W_f1 = tf.Variable(tf.random_normal([1024*4*4, 2048], stddev = 0.01))
b_f1 = tf.Variable(tf.random_normal([2048]))

L_f1 = tf.nn.relu(tf.matmul(L_flat, W_f1) + b_f1)

# FC 2
W_f2 = tf.Variable(tf.random_normal([2048, 1024], stddev = 0.01))
b_f2 = tf.Variable(tf.random_normal([1024]))

L_f2 = tf.nn.relu(tf.matmul(L_f1, W_f2) + b_f2)

# FC 3
W_f3 = tf.Variable(tf.random_normal([1024, 10], stddev = 0.01))
b_f3 = tf.Variable(tf.random_normal([10]))

logits = tf.nn.relu(tf.matmul(L_f2, W_f3) + b_f3)

# softmax
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
# optimizer
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)

# initialize
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())
#step = 1

# train
print('AlexNet Learning started. It takes sometime.')
for epoch in range(training_epochs):
  # initialize
  avg_cost = 0
  total_batch = int(mnist.train.num_examples / batch_size)

  for i in range(total_batch):
    batch_xs, batch_ys = mnist.train.next_batch(batch_size)
    feed_dict = {X: batch_xs, Y: batch_ys, keep_prob: 0.7}
    c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
    avg_cost += c / total_batch

#    if step % display_step == 0:
  print('Epoch:', '%04d' % (epoch + 1), 'cost=', '{:.9f}'.format(avg_cost))

#    step += 1

print('Learning Finished!')
