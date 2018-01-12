from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten
from keras.models import Model
from keras import backend as K

def build_cnn():
    input_img = Input(shape = (28, 28, 1))

    x = Conv2D(16, (3, 3), activation='relu', padding='same', name = 'c1')(input_img)
    x = MaxPooling2D((2, 2), padding='same', name = 'm1')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same', name = 'c2')(x)
    x = MaxPooling2D((2, 2), padding='same', name = 'm2')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same', name = 'c3')(x)
    encoded = MaxPooling2D((2, 2), padding='same', name = 'm3')(x)
    
    # at this point the representation is (4, 4, 8) i.e. 128-dimensional
    
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation='relu')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
    
    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    return autoencoder

def build_dnn():
    input_img = Input(shape = (784,))
    encoded = Dense(128, activation='relu')(input_img)
    encoded = Dense(64, activation='relu')(encoded)
    encoded = Dense(32, activation='relu')(encoded)
    
    decoded = Dense(64, activation='relu')(encoded)
    decoded = Dense(128, activation='relu')(decoded)
    decoded = Dense(784, activation='sigmoid')(decoded)

    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

    return autoencoder

def build_dnn2():
    input_img = Input(shape = (784,))

    encoded = Dense(1024, activation='relu', name='e1')(input_img)
    encoded = Dense(512, activation='relu', name='e2')(encoded)
    encoded = Dense(256, activation='relu', name='e3')(encoded)
    encoded = Dense(128, activation='relu', name='e4')(encoded)

    encoded = Dense(64, activation='relu', name='e5')(encoded)

    decoded = Dense(128, activation='relu')(encoded)
    decoded = Dense(256, activation='relu')(decoded)
    decoded = Dense(512, activation='relu')(decoded)
    decoded = Dense(1024, activation='relu')(decoded)

    decoded = Dense(784, activation='sigmoid')(decoded)
    

    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

    return autoencoder
    

# def cnn_encoder(weight_path):
#     input_img = Input(shape = (28, 28, 1))

#     x = Conv2D(16, (3, 3), activation='relu', padding='same', name = 'c1')(input_img)
#     x = MaxPooling2D((2, 2), padding='same', name = 'm1')(x)
#     x = Conv2D(8, (3, 3), activation='relu', padding='same', name = 'c2')(x)
#     x = MaxPooling2D((2, 2), padding='same', name = 'm2')(x)
#     x = Conv2D(8, (3, 3), activation='relu', padding='same', name = 'c3')(x)
#     encoded = MaxPooling2D((2, 2), padding='same', name = 'm3')(x)
#     out = Flatten()(encoded)

#     model = Model(input_img, out)
#     model.load_weights(weight_path, by_name=True)

#     return model

def cnn_encoder(weight_path):
    input_img = Input(shape = (28, 28, 1))

    x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same', name = 'c3')(x)
    encoded = MaxPooling2D((2, 2), padding='same', name = 'm3')(x)
    out = Flatten()(encoded)

    model = Model(input_img, out)
    model.load_weights(weight_path, by_name=True)

    return model


def dnn_encoder(model):
    input_img = Input(shape=(784,), name = 'input_0')
    encoded = model.layers[0](input_img)
    encoded = model.layers[1](encoded)
    encoded = model.layers[2](encoded)

    model = Model(input_img, encoded)
    return model
    
def dnn_encoder2(weight_path):
    input_img = Input(shape = (784,))
    encoded = Dense(1024, activation='relu', name='e1')(input_img)
    encoded = Dense(512, activation='relu', name='e2')(encoded)
    encoded = Dense(256, activation='relu', name='e3')(encoded)
    encoded = Dense(128, activation='relu', name='e4')(encoded)

    encoded = Dense(64, activation='relu', name='e5')(encoded)

    model = Model(input_img, encoded)
    model.load_weights(weight_path, by_name=True)

    return model