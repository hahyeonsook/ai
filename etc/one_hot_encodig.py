token = ['나', '는', '자연어', '처리', '를', '배운다']

print(token)

word2index={}  # dictionary로 선언, 단어 값이 key/ index값이 value
# []는 key에 대응하는 value를 할당하거나 value에 접근할 때 쓰인다.

for voca in token: # token list의 모든 값을 for
  if voca not in word2index.keys():
    word2index[voca]=len(word2index)  # len(word2index)의 초기값은 0이므로 voca key에 0의 value를 할당해줄 수 있음

print(word2index)

def one_hot_encoding(word, word2index):
  one_hot_vector=[0]*(len(word2index))
  index = word2index(word)
  one_hot_vector[index] = 1
  return one_hot_vector


