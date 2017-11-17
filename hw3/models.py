import os

from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Flatten, Activation, Conv2D, Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization 
from keras.optimizers import SGD, Adam, Adadelta

def build_model(nb_class = 7, dropout_conv = 0.5, dropout_fully = 0.25):
    '''
    # Returns 
    Keras model
    '''
    print('\nBuilding model...')
    model = Sequential()
    # CNN
    #b0
    model.add(Conv2D(64, (5, 5),
            border_mode ='same',
            input_shape = (48, 48, 1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Dropout(dropout_conv))

    # b1
    model.add(Conv2D(128, (3, 3), # 有修改過, 上次 best 是 64
            border_mode ='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(dropout_conv))

    model.add(Conv2D(128,(3, 3),
            border_mode ='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Dropout(dropout_conv))
    #b2
    model.add(Conv2D(256,(3, 3),
            border_mode ='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(dropout_conv))

    model.add(Conv2D(256,(3, 3),
            border_mode ='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Dropout(dropout_conv))
    #b3
    model.add(Conv2D(512,(3, 3),
            border_mode ='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(dropout_conv))

    model.add(Conv2D(512,(3, 3),
            border_mode ='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Dropout(dropout_conv))

    # Flatten
    model.add(Flatten())

    model.add(Dense(1024))
    model.add(BatchNormalization())    
    model.add(Activation('relu'))
    model.add(Dropout(dropout_fully))

    model.add(Dense(1024))
    model.add(BatchNormalization())    
    model.add(Activation('relu'))
    model.add(Dropout(dropout_fully))

    model.add(Dense(nb_class))  
    model.add(Activation('softmax'))

    # sgd = SGD(lr = 0.01, decay = 0.0)

    model.compile(loss = 'categorical_crossentropy', 
                optimizer = 'adam', 
                metrics = ['accuracy'])
    model.summary()
    return model

def build_dnn(dropout = 0.2, nb_class = 7):
    model  = Sequential()

    model.add(Flatten(input_shape = (48, 48, 1)))
    # model.add(Dense(147456))
    # model.add(BatchNormalization())
    # model.add(Activation('relu'))
    # model.add(Dropout(dropout))

    # model.add(Dense(36864))
    # model.add(BatchNormalization())
    # model.add(Activation('relu'))
    # model.add(Dropout(dropout))

    # model.add(Dense(18432))
    # model.add(BatchNormalization())
    # model.add(Activation('relu'))
    # model.add(Dropout(dropout))

    model.add(Dense(9261))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(dropout))

    model.add(Dense(4608))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(dropout))

    model.add(Dense(1024))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(dropout))

    model.add(Dense(1024))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(dropout))

    model.add(Dense(nb_class))  
    model.add(Activation('softmax'))


    model.compile(loss = 'categorical_crossentropy', 
                optimizer = 'adam', 
                metrics = ['accuracy'])
    model.summary()

    return model

def build_xception(input_shape, num_classes):
    img_input = Input(input_shape)
    x = Conv2D(32, (3, 3), strides=(2, 2), use_bias=False)(img_input)
    x = BatchNormalization(name='block1_conv1_bn')(x)
    x = Activation('relu', name='block1_conv1_act')(x)
    x = Conv2D(64, (3, 3), use_bias=False)(x)
    x = BatchNormalization(name='block1_conv2_bn')(x)
    x = Activation('relu', name='block1_conv2_act')(x)

    residual = Conv2D(128, (1, 1), strides=(2, 2),
                      padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = SeparableConv2D(128, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization(name='block2_sepconv1_bn')(x)
    x = Activation('relu', name='block2_sepconv2_act')(x)
    x = SeparableConv2D(128, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization(name='block2_sepconv2_bn')(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])

    residual = Conv2D(256, (1, 1), strides=(2, 2),
                      padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = Activation('relu', name='block3_sepconv1_act')(x)
    x = SeparableConv2D(256, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization(name='block3_sepconv1_bn')(x)
    x = Activation('relu', name='block3_sepconv2_act')(x)
    x = SeparableConv2D(256, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization(name='block3_sepconv2_bn')(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])
    x = Conv2D(num_classes, (3, 3),
            #kernel_regularizer=regularization,
            padding='same')(x)
    x = GlobalAveragePooling2D()(x)
    output = Activation('softmax',name='predictions')(x)

    model = Model(img_input, output)

#     opt = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
#     opt = Adam(lr=1e-3, beta_1=0.81, beta_2=0.899)
#     opt = Adadelta(lr=0.1, rho=0.95, epsilon=1e-08)

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    return model

def build_ta_model(input_shape = (48, 48, 1), num_classes = 7):
    input_img = Input(input_shape)
    
    block1 = Conv2D(64, (5, 5), padding='valid', activation='relu')(input_img)
    block1 = ZeroPadding2D(padding=(2, 2), data_format='channels_last')(block1)
    block1 = MaxPooling2D(pool_size=(5, 5), strides=(2, 2))(block1)
    block1 = ZeroPadding2D(padding=(1, 1), data_format='channels_last')(block1)

    block2 = Conv2D(64, (3, 3), activation='relu')(block1)
    block2 = ZeroPadding2D(padding=(1, 1), data_format='channels_last')(block2)

    block3 = Conv2D(64, (3, 3), activation='relu')(block2)
    block3 = AveragePooling2D(pool_size=(3, 3), strides=(2, 2))(block3)
    block3 = ZeroPadding2D(padding=(1, 1), data_format='channels_last')(block3)

    block4 = Conv2D(128, (3, 3), activation='relu')(block3)
    block4 = ZeroPadding2D(padding=(1, 1), data_format='channels_last')(block4)

    block5 = Conv2D(128, (3, 3), activation='relu')(block4)
    block5 = ZeroPadding2D(padding=(1, 1), data_format='channels_last')(block5)
    block5 = AveragePooling2D(pool_size=(3, 3), strides=(2, 2))(block5)
    block5 = Flatten()(block5)

    fc1 = Dense(1024, activation='relu')(block5)
    fc1 = Dropout(0.2)(fc1)

    fc2 = Dense(1024, activation='relu')(fc1)
    fc2 = Dropout(0.2)(fc2)

    predict = Dense(num_classes)(fc2)
    predict = Activation('softmax')(predict)
    model = Model(inputs=input_img, outputs=predict)

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model