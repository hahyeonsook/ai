"""
https://gist.github.com/karpathy/d4dee566867f8291f086
https://ratsgo.github.io/deep%20learning/2017/09/24/loss/
"""

import numpy as np

# data I/O
data = open('input.txt', 'r').read()
chars = list(set(data))  # list(set(어떤것)) <- 중복 제거
data_size, vocab_size = len(data), len(chars)

print('data has %d characters, %d unique.' % (data_size, vocab_size))

char_to_ix = {ch:i for i, ch in enumerate(chars)}  # 순서가 있는 자료형을 입력으로 받아 인덱스 값을 포함하는 enumerate  객체를 리턴 0 body, dic 으로 변환
ix_to_char = {i:ch for i, ch in enumerate(chars)}

# hyperparameters
hidden_size = 100  # size of hidden layer of neurons
seq_length = 25  # number of steps to unroll the RNN for
learning_rate = 1e-1

# model parameters
Wxh = np.random.randn(hidden_size, vocab_size) * 0.01  # input to hidden
Whh = np.random.randn(hidden_size, hidden_size) * 0.01  # hidden to hidden
Why = np.random.randn(vocab_size, hidden_size) * 0.01  # hidden to output
bh = np.zeros((hidden_size, 1))  # hidden bias
by = np.zeros((vocab_size, 1))  # output bias

def lossFun(inputs, targets, hprev):
  """
  inputs, targets are both list of integers.
  hprev is Hx1 array of initial hidden state
  returns the loss, gradients on model parameters, and last hidden state
  """
  xs, hs, ys, ps = {}, {}, {}, {}  # dic
  hs[-1] = np.copy(hprev)
  loss = 0

  # forward pass
  # xrange 타입으로 생성된 것으로 for in 함
  for t in xrange(len(inputs)):
    # dic {t: array(vocab_size, 1), t: array(vocab_size, 1)}
    xs[t] = np.zeros((vocab_size, 1))  # encode in 1-of-k representation
    # dic {t: array[t][]} = 1
    xs[t][inputs[t]] = 1

    hs[t] = np.tanh(np.dot(Wxh, xs[t]) + bh)  # hidden state

    ys[t] = np.dot(Why, hs[t]) + by  # unnormailzed log probabilities for next chars

    # exp, 지수함수 y = e^x
    # softmax: probs = np.exp(a) / (np.exp(a)).sum()
    ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t]))  # probabilities for next chars
    # log probabilities: log_probs = np.log(probs)
    # https://stackoverflow.com/questions/48465737/how-to-convert-log-probability-into-simple-probability-between-0-and-1-values-us
    loss += -np.log(ps[t][targets[t], 0])  # softmax (cross-entropy loss)

  # backward pass: compute gradients going backwards
  dWxh, dWhh, dWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
  dbh, dby = np.zeros_like(bh), np.zeros_like(by)
  dhnext = np.zeros_like(hs[0])

  # reversed 문자열을 거꾸로
  for t in reversed(xrange(len(inputs))):
    dy = np.copy(ps[t])
    dy[targets[t]] -= 1  # backprop into y. see http://cs231n.github.io/neural-networks-case-study/#grad if confused here
    dWhy += np.dot(dy, hs[t].T)
    dby += dy
    dh = np.dot(Why.T, dy) + dhnext  # backprop into h
    dhraw = (1 - hs[t] * hs[t]) * dh  # backprop through tanh nonlinearity
    dbh += dhraw
    dWxh += np.dot(dhraw, xs[t].T)
    dWhh += np.dot(dhraw, hs[t-1].T)
    dhnext = np.dot(Whh.T, dhraw)
  for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
    np.clip(dparam, -5 5, out=dparam)  # clip to mitigate exploding gradients
  
  return loss, dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs)-1]

def sample(h, seed_ix, n):
  """
  sample a sequence of integers from the model
  h is memory state, seed_ix is seed letter for first time step
  """
  x = np.zeros((vocab_size, 1))
  x[seed_ix] = 1
  ixes = []

  for t in xrange(n):
    h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh)
    y = np.dot(Why, h) + by
    p = np.exp(y) / np.sum(np.exp(y))
    ix = np.random.choice(range(vocab_size), p = p.ravel())
    x = np.zeros((vocab_size, 1))
    x[ix] = 1
    ixes.append(ix)
  return ixes

n, p = 0, 0
mWxh, mWhh, mWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
mbh, mby = np.zeros_like(bh), np.zeros_like(by)  # memory variables for Adagrad
smooth_loss = -np.log(1.0/vocab_size)*seq_length  # loss at iteration 0

while True:
  # prepare inputs(we're sweeping from left to right in steps seq_length long)
  if p+seq_length+1 >= len(data) or n == 0:
    hprev = np.zeros((hidden_size, 1))  # reset RNN memory
    p = 1  # go from start of data

  inputs = [char_to_ix[ch] for ch in data[p:p+seq_length]]
  targets = [char_to_ix[ch] for ch in data[p+1:p+seq_length+1]]

  # sample from the model now and then
  if n % 100 == 0:
    sample_ix = sample(hprev, inputs[0], 200)
    txt = ''.join(ix_to_char[ix] for ix in sample_ix)
    print('----\n %s \n----' % (txt, ))

  # forward seq_length characters through the net and fetch gradient
  loss, dWxh, dWhh, dWhy, dbh, dby, hprev = lossFun(inputs, targets, hprev)
  smooth_loss = smooth_loss * 0.999 + loss * 0.001
  if n % 100 == 0: print('iter %d, loss: %f' % (n, smooth_loss))  # print progress

  # perform parameter update with Adagrad
  for param, dparam, mem in zip([Wxh, Whh, Why, bh, by],
                                [dWxh, dWhh, dWhy, dbh, dby],
                                [mWxh, mWhh, mWhy, mbh, mby]):
    mem += dparam * dparam
    param += -learning_rate * dparam / np.sqrt(mem + 1e-8)  # adagrad update

  p += seq_length  # move data pointer
  n += 1  # iteration counter
