import numpy as np
import keras.backend as kb
from keras import optimizers
from keras.layers import Input, Embedding, Flatten, Dot, Add, Dense, Dropout, BatchNormalization, Concatenate, Activation, Reshape
from keras.models import Model
from keras.regularizers import l2


def build_mf(num_usr, num_movie):
    DIM = 256

    user_input = Input(shape = (1,))
    movie_input = Input(shape = (1,))

    user_vec = Embedding(input_dim = num_usr, 
            output_dim = DIM,
            embeddings_regularizer = l2(0.000005),
            input_length = 1)(user_input)
    user_vec = Flatten()(user_vec)

    movie_vec = Embedding(input_dim = num_movie, 
            output_dim = DIM, 
            embeddings_regularizer = l2(0.000005),
            input_length = 1)(movie_input)
    movie_vec = Flatten()(movie_vec)

    user_bias = Embedding(input_dim = num_usr, 
            output_dim = 1, 
            embeddings_initializer='zero')(user_input)
    user_bias = Flatten()(user_bias)

    movie_bias = Embedding(input_dim = num_movie,
            output_dim = 1,
            embeddings_initializer='zero')(movie_input)
    movie_bias = Flatten()(movie_bias)
  

    matrix_dot = Dot(axes = 1)([user_vec, movie_vec])
    matrix_add = Add()([matrix_dot, user_bias, movie_bias])

    model = Model([user_input, movie_input], matrix_add)
    model.summary()

    opt = optimizers.Adam(lr=0.0005)
    model.compile(optimizer = opt, loss = 'mse', metrics = [rmse])

    return model

def build_nn(num_usr, num_movie):
    DIM = 256
    DEPTH = 3
    DROP = 0.5

    user_input = Input(shape = (1,))
    movie_input = Input(shape = (1,))

    user_vec = Embedding(input_dim = num_usr, 
            output_dim = DIM,
        #     embeddings_regularizer = l2(0.000005),
            input_length = 1)(user_input)
    user_vec = Flatten()(user_vec)

    movie_vec = Embedding(input_dim = num_movie, 
            output_dim = DIM, 
        #     embeddings_regularizer = l2(0.000005),
            input_length = 1)(movie_input)
    movie_vec = Flatten()(movie_vec)
    
    nn = Concatenate()([user_vec, movie_vec])

    for i in range(DEPTH):
        width = int(256 / (2 ** i))
        if width < 2:
            width = 2
        nn = Dense(width, activation='relu')(nn)
        nn = Dropout(DROP)(nn)
    output = Dense(1, activation='linear')(nn)
    model = Model([user_input, movie_input], output)
    model.summary()
    opt = optimizers.Adam(lr=0.0005)
    model.compile(optimizer = opt, loss = 'mse', metrics = [rmse])

    return model

def rmse(y, y_pred):
    return kb.sqrt(kb.mean((y_pred-y) ** 2))