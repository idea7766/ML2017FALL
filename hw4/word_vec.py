import utils
import sys
from gensim.models import Word2Vec as w2v
from gensim import corpora
from keras.preprocessing import text

# set parameters

## data path
TRAIN = './data/training_label.txt'
TRAIN_NO_LAB = 'data/training_nolabel.txt'
TEST = 'data/testing_data.txt'
# TRAIN = sys.argv[1]
# TRAIN_NO_LAB = sys.argv[2]
# TEST = sys.argv[3]

## parameters of word2vec
WINDOW = 10
VEC_DIM = 100

# load (label, setence)
y_train, x_train = utils.load_data(TRAIN, file_type = 'train')
x_train_nolab = utils.load_data(TRAIN_NO_LAB, file_type = 'train_nolabel')
_, x_test = utils.load_data(TEST, file_type = 'test')

setence = x_train + x_train_nolab + x_test

# setence to word list e.g. 'fxxk you' to ['fxxk', 'you']
setenece_split = []
for line in setence:
    words = text.text_to_word_sequence(line, filters='!"\'#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
    setenece_split.append(words)
    
# print(setenece_split[0:10]) # just for test
dic = corpora.Dictionary(setenece_split)
dic.save('./dictionary')
print(dic)

# word to vector
model = w2v(setenece_split, window = WINDOW, size = VEC_DIM)
model.save('./word_vec')

# print(model.most_similar('get'))
# print(model.similarity('get', 'getting'))