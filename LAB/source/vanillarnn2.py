import numpy as np

# 1. Data I/O
data = open('input.txt', 'r').read()
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)

# 2. Data encoding
char_to_ix = {ch:i for i, ch in enumerate(chars)}
ix_to_char = {i:ch for i, ch in enumerate(chars)}

# hyper parameters
hidden_size = 100
seq_length = 25
learning_rate = 1e-1

# model parameters
Wxh = np.random.randn(hidden_size, vocab_size) * 0.01
Whh = np.random.randn(hidden_size, hidden_size) * 0.01
Wyh = np.random.randn(vocab_size, hidden_size) * 0.01
bh = np.zeros((hidden_size, 1))
by = np.zeros((vocab_size, 1))

# 3. loss func
def lossFunc():
  xs, hs, ys, ps = {}, {}, {}, {}
  hs[-1] = np.copy()
  loss = 0

  # forward pass
  for t in xrange(len(inputs)):
    xs[t] = np.zeros((vocab_size, 1))
    xs[t][inputs[t]] = 1
