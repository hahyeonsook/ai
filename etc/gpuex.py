import numpy as np
import tensorflow as tf
import datetime

A = np.random.rand(1e4, 1e4).astype('float32')
B = np.random.rand(1e4, 1e4).astype('float32')

n = 10

c1 = []
c2 = []

def matpow(M, n):
    if n < 1: #Abstract cases where n &lt; 1
       return M
    else:
       return tf.matmul(M, matpow(M, n-1))

with tf.device('/gpu:0'):
    a = tf.constant(A)
    b = tf.constant(B)
    c1.append(matpow(a, n))
    c1.append(matpow(b, n))

with tf.device('/cpu:0'):
    sum = tf.add_n(c1)

t1_1 = datetime.datetime.now()

with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    sess.run(sum)
t2_1 = datetime.datetime.now()


print("Single GPU computation time: " + str(t2_1-t1_1))
print("Multi GPU computation time: " + str(t2_2-t1_2))
