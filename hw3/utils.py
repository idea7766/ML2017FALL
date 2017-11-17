import numpy as np

# from keras.callbacks import EarlyStopping
from keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical

import models

def self_learn(pred_model, data, x_train, y_train,nb_class, min_proba):
    '''
    # Arguments
    pred_model : 用來預測 data 的 model
    data : 被預測的 data 
    nb_class : class 的數量
    min_proba : 預測後資料的機率 > min_proba 再拿來延伸 traning set 做 traning 
    # Return
    keras model
    '''
    TRAIN_FOLDER = './re0'
    CHECK_POINT_NAME = TRAIN_FOLDER + '/model.{epoch:04d}-{val_acc:.4f}.h5'
    LOG_NAME = TRAIN_FOLDER + '/log.csv'
    PATIENCE = 50

    result = pred_model.predict_proba(data)

    feats = []
    lables = []

    len_result = len(result)
    for ind, col in zip(range(len_result),result):
        i = np.argmax(col)
        if col[i] >= min_proba:
            class_col = np.zeros(nb_class)
            class_col[i] = 1
            lables.append(class_col)
            feats.append(data[ind])
    
    feats = np.asarray(feats)
    lables = np.asarray(lables)

    X_train ,Y_train ,X_val, Y_val = validation(x_train, y_train)
    feats, lables, feats_val, lables_val = validation(feats, lables)

    X_train = np.concatenate((X_train, feats), axis=0)
    Y_train = np.concatenate((Y_train, lables), axis=0)
    X_val = np.concatenate((X_val, feats_val), axis=0)
    Y_val = np.concatenate((Y_val, lables_val), axis=0)
            

    callback = [
            TensorBoard(),
            CSVLogger(LOG_NAME, append=True),
            ModelCheckpoint(CHECK_POINT_NAME, period=10),
            ReduceLROnPlateau('val_loss', factor=0.1, patience=int(PATIENCE/4), verbose=1),
            EarlyStopping(patience = PATIENCE)
            ]
    
    train_gen = generator(X_train, Y_train, batch_size = 128)

    model = models.build_model(dropout_conv = 0.2, dropout_fully = 0.2)

    model.fit_generator(
        train_gen,
        samples_per_epoch = X_train.shape[0],    
        epochs = 1000,
        validation_data = (X_val, Y_val),
        callbacks=callback
    )
    return model

def generator(x, y, batch_size = 256):

    img_gen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    gen = img_gen.flow(
        x,
        y,
        batch_size = batch_size
    )
    
    return gen

def to_nb_class(lables):
    nb_lables = []
    length = len(lables)
    for col in lables:
        nb_class = np.argmax(col)
        nb_lables.append(nb_class)

    nb_lables = np.asarray(nb_lables)

    return nb_lables
def validation(feats, lables, ratio = 0.1):
    '''
    # Arguments
    reatio : 多少比例的資料要被當作 validation set 
    # Return
    X_train, Y_train, X_val, Y_val
    '''
    nb_cut = int(np.shape(feats)[0] *ratio)

    X_train = feats[:-nb_cut]
    Y_train = lables[:-nb_cut]

    X_val = feats[-nb_cut:]
    Y_val = lables[-nb_cut:]

    return X_train, Y_train, X_val, Y_val