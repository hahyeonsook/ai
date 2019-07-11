"""
https://gist.github.com/karpathy/d4dee566867f8291f086
https://ratsgo.github.io/deep%20learning/2017/09/24/loss/
https://ratsgo.github.io/deep%20learning/2017/10/02/softmax/
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
seq_length = 25  # RNN 단계
learning_rate = 1e-1

# model parameters
Wxh = np.random.randn(hidden_size, vocab_size) * 0.01  # input to hidden
Whh = np.random.randn(hidden_size, hidden_size) * 0.01  # hidden to hidden
Why = np.random.randn(vocab_size, hidden_size) * 0.01  # hidden to output
bh = np.zeros((hidden_size, 1))  # hidden bias
by = np.zeros((vocab_size, 1))  # output bias

def lossFun(inputs, targets, hprev):
  """
  [inputs] 숫자와 단어를 짝지은 dic(char_to_ix)로, input받은 데이터를 숫자로 변환시킨 것
  [targets] input받은 글을 첫문자를 빼놓고 숫자로 변환시킨 것, rnn은 다음 글자를 예상하는 nn이므로 정답임
  inputs, targets are both list of integers.
  hprev is Hx1 array of initial hidden state list(100, 1)
  returns the loss, gradients on model parameters, and last hidden state
  """
  xs, hs, ys, ps = {}, {}, {}, {}  # dic
  # hs dictionary의 끝에 (100, 1) list of integers를 넣음
  hs[-1] = np.copy(hprev)
  loss = 0

  # forward pass
  # xrange 타입으로 생성된 것으로 for in 함
  # 문자열을 순서대로 돌림
  for t in range(len(inputs)):
    # dic {t: array(vocab_size, 1), t: array(vocab_size, 1)}
    # t번째 {}의 array를 (문자종류, 1)로 초기화한 후
    xs[t] = np.zeros((vocab_size, 1))  # encode in 1-of-k representation
    # dic {t: array[t][]} = 1
    # t번째 {}의 array에서 inputs[t](=array의 행번호)값의 번호를 1로 ex)inputs[1]=2, array[[0], [0], [1], ...]
    xs[t][inputs[t]] = 1

    hs[t] = np.tanh(np.dot(Wxh, xs[t]) + bh)  # hidden state

    ys[t] = np.dot(Why, hs[t]) + by  # next chars의 예상

    # exp, 지수함수 y = e^x
    # softmax: probs = np.exp(a) / (np.exp(a)).sum()
    ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t]))  # next chars의 확률들
    # log probabilities: log_probs = np.log(probs)
    # https://stackoverflow.com/questions/48465737/how-to-convert-log-probability-into-simple-probability-between-0-and-1-values-us
    loss += -np.log(ps[t][targets[t], 0])  # softmax (cross-entropy loss), loss값을 cross-entropy로 계산

  # backward pass: backwards하는 gradients를 계산함
  dWxh, dWhh, dWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)  # 모양이 같고, 0으로 초기화된 list를 생성함
  dbh, dby = np.zeros_like(bh), np.zeros_like(by)
  dhnext = np.zeros_like(hs[0])

  # reversed, 문자열의 길이를 range, list의 끝번호부터
  for t in reversed(xrange(len(inputs))):
    dy = np.copy(ps[t])
    # 
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
    # np.clip(배열, 최소값 기준, 최대값 기준): 최소값과 최대값 조건으로 이 기준을 벗어나는 값에 대해서는 일괄적으로 최소값, 최대값으로 대치
    np.clip(dparam, -5, 5, out=dparam)  # clip to mitigate exploding gradients
  
  return loss, dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs)-1]

#########################################

def sample(h, seed_ix, n):
  """
  sample a sequence of integers from the model
  h is memory state, seed_ix is seed letter for first time step
  """
  x = np.zeros((vocab_size, 1))
  x[seed_ix] = 1
  ixes = []

  for t in range(n):
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
# (1.0/존재하는 문자의 개수)*
smooth_loss = -np.log(1.0/vocab_size)*seq_length  # loss at iteration 0

while True:
  # prepare inputs(we're sweeping from left to right in steps seq_length long)
  if p+seq_length+1 >= len(data) or n == 0:
    hprev = np.zeros((hidden_size, 1))  # reset RNN memory
    p = 1  # go from start of data

  # input 문자를 번호를 encoding하고, 정답 문자도 encoding함
  inputs = [char_to_ix[ch] for ch in data[p:p+seq_length]]
  targets = [char_to_ix[ch] for ch in data[p+1:p+seq_length+1]]

  # model에서 sample을 추출
  if n % 100 == 0:
    sample_ix = sample(hprev, inputs[0], 200)
    # list를 문자열로 변환
    txt = ''.join(ix_to_char[ix] for ix in sample_ix)
    print('----\n %s \n----' % (txt, ))

  # net and fetch gradient을 통해 seq_length characters을 forwards
  loss, dWxh, dWhh, dWhy, dbh, dby, hprev = lossFun(inputs, targets, hprev)
  smooth_loss = smooth_loss * 0.999 + loss * 0.001
  if n % 100 == 0: print('iter %d, loss: %f' % (n, smooth_loss))  # print progress

  # Adagrad로 parameter update
  # zip 동일한 개수로 이루어진 자료형을 묶어주는 역할
  for param, dparam, mem in zip([Wxh, Whh, Why, bh, by],
                                [dWxh, dWhh, dWhy, dbh, dby],
                                [mWxh, mWhh, mWhy, mbh, mby]):
    mem += dparam * dparam
    param += -learning_rate * dparam / np.sqrt(mem + 1e-8)  # adagrad update

  p += seq_length  # move data pointer
  n += 1  # iteration counter
