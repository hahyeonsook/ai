import numpy as np

# lossFunc(입력(int list), 정답(int list), hidden state)
def lossFunc(inputs, targets, hprev):
  xs, hs, ys, ps = {}, {}, {}, {}
  hs[-1] = np.copy(hprev)
  loss = 0

  # forward pass
  for t in range(len(inputs)):
    # dict[t]: t번째 문자의 one hot array를 0으로 초기화
    xs[t] = np.zeros((vocab_size, 1))
    # dict[t]: t번째 문자의 encoding number에 맞게 one hot 해줌
    xs[t][inputs[t]] = 1
    # dict[t]: t번째 hidden state(100, 1)값 계산. tanh(Wxh * xt + bh)
    hs[t] = np.tanh(np.dot(Wxh, xs[t]) + bh)
    # dict[t]: t번째 output값 계산(next char 예상 값). Why * ht + by
    ys[t] = np.dot(Why, hs[t]) + by
    # dict[t]: t번째 문자가 각 문자에 해당될 가능성 array를 확률 값으로 변경. softmax
    ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t]))
    # dict[t]: t번째 문자의 확률 array(ps[t])에서 (정답 번호대로 나열)정답이 될 확률 손실 계산
    # cross-entropy(확률분포의 차이계산)
    loss += -np.log(ps[t][targets[t], 0])

  # backward pass
  dWxh, dWhh, dWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
  dbh, dby = np.zeros_like(bh), np.zeros_like(by)
  # 반드시 존재하는 hs[0]값 모양대로 0으로 초기화 (100, 1)
  dhnext = np.zeros_like((hs[0]))

  for t in reversed(range(len(inputs))):
    # 예상 array를 copy
    dy = np.copy(ps[t])
    # 정답일 확률로 예측한 값에서 -1
    dy[targets[t]] -= 1  # backprop into y
    # dict[t]: t번째 문자의 hidden state와 t+1번째 문자 예상 확률을 dot하고 Why에 +
    dWhy += np.dot(dy, hs[t].T)
    dby += dy
    dh = np.dot(Why.T, dy) + dhnext  # backprop into h
    dhraw = (1 - hs[t] * hs[t]) * dh  # backprop through tanh nonlinearity
    dbh += dhraw
    dWxh += np.dot(dhraw, xs[t].T)
    dWhh += np.dot(dhraw, hs[t-1].T)
    dhnext = np.dot(Whh.T, dhraw)

  for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
    np.clip(dparam, -5, 5, out=dparam)  # clip to mitigate exploding gradients

  return loss, dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs)-1]
