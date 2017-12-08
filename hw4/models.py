from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers.recurrent import GRU, LSTM
from keras.layers.embeddings import Embedding
from keras.layers.wrappers import Bidirectional
import utils

def build_GRU(num_word, num_class, embedding_matrix, emb_dim = 100, drop_rate = 0.1):
    model = Sequential()
    # embedding sequence
    model.add(Embedding(num_word, emb_dim, weights = embedding_matrix, trainable=False))

    model.add(Bidirectional(GRU(256, return_sequences=True)))
    model.add(Bidirectional(GRU(256)))    
    model.add(Dropout(drop_rate))    
    # model.add(LSTM(128))
    # model.add(Dropout(drop_rate))
    
    
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(drop_rate))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(drop_rate))

    model.add(Dense(512, activation='relu'))
    model.add(Dropout(drop_rate))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(drop_rate))

    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(drop_rate))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(drop_rate))

    model.add(Dense(num_class, activation='sigmoid'))
    
    model.summary()
    model.compile(optimizer = 'adam',
                  loss = 'binary_crossentropy',
                  metrics = ['accuracy'])
    return model

def build_LSTM(num_word, num_class, embedding_matrix, emb_dim = 100, drop_rate = 0.1):
    model = Sequential()
    # embedding sequence
    model.add(Embedding(num_word, emb_dim, weights = embedding_matrix, trainable=False))

    model.add(Bidirectional(LSTM(256, return_sequences=True)))
    model.add(Bidirectional(LSTM(256)))    
    model.add(Dropout(drop_rate))        
    
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(drop_rate))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(drop_rate))

    model.add(Dense(512, activation='relu'))
    model.add(Dropout(drop_rate))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(drop_rate))

    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(drop_rate))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(drop_rate))

    model.add(Dense(num_class, activation='sigmoid'))
    
    model.summary()
    model.compile(optimizer = 'adam',
                  loss = 'binary_crossentropy',
                  metrics = ['accuracy'])
    return model

def build_simpleGRU(num_word, num_class, embedding_matrix, emb_dim = 100, drop_rate = 0.1):
    model = Sequential()
    # embedding sequence
    model.add(Embedding(num_word, emb_dim, weights = embedding_matrix, trainable=False))

    model.add(GRU(128))
    model.add(Dropout(drop_rate))
    
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(drop_rate))

    model.add(Dense(256, activation='relu'))
    model.add(Dropout(drop_rate))

    model.add(Dense(512, activation='relu'))
    model.add(Dropout(drop_rate))

    model.add(Dense(num_class, activation='sigmoid'))
    
    model.summary()
    model.compile(optimizer = 'adam',
                  loss = 'binary_crossentropy',
                  metrics = ['accuracy'])
    return model

def build_simpleLSTM(num_word, num_class, embedding_matrix, emb_dim = 100, drop_rate = 0.1):
    model = Sequential()
    # embedding sequence
    model.add(Embedding(num_word, emb_dim, weights = embedding_matrix, trainable=False))

    model.add(LSTM(128))
    model.add(Dropout(drop_rate))    
    
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(drop_rate))

    model.add(Dense(256, activation='relu'))
    model.add(Dropout(drop_rate))

    model.add(Dense(512, activation='relu'))
    model.add(Dropout(drop_rate))

    model.add(Dense(num_class, activation='sigmoid'))
    
    model.summary()
    model.compile(optimizer = 'adam',
                  loss = 'binary_crossentropy',
                  metrics = ['accuracy'])
    return model

def build_DNN(input_dim, num_class = 1, drop_rate = 0.1):
    model = Sequential()
    
    model.add(Dense(256, input_dim=input_dim, activation='relu'))
    model.add(Dropout(drop_rate))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(drop_rate))

    model.add(Dense(512, activation='relu'))
    model.add(Dropout(drop_rate))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(drop_rate))

    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(drop_rate))
    model.add(Dense(1024, activation='relu'))

    model.add(Dense(num_class, activation='sigmoid'))
    
    model.summary()
    model.compile(optimizer = 'adam',
                  loss = 'binary_crossentropy',
                  metrics = ['accuracy'])
    return model