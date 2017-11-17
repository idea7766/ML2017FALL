import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from keras.models import Sequential

import read_n_write as rw
import utils
import models

base_dir = os.path.dirname(__file__)
TRAIN_FOLDER = './model_chk/dnn1'
CHECK_POINT_NAME = TRAIN_FOLDER + '/model.{epoch:04d}-{val_acc:.4f}.h5'
LOG_NAME = TRAIN_FOLDER + '/log.csv'
PATIENCE = 50

feats, lables, _ = rw.read_dataset('data/train.csv', shuffle = True)
X_train, Y_train, X_val, Y_val = utils.validation(feats, lables, 0.1)

train_data_gen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
)
train_data_gen.fit(X_train)

val_data_gen = ImageDataGenerator()

# early_stopping = EarlyStopping(patience = 50)
callback = [
            TensorBoard(),
            CSVLogger(LOG_NAME, append=True),
            ModelCheckpoint(CHECK_POINT_NAME, period=10),
            ReduceLROnPlateau('val_loss', factor=0.1, patience=int(PATIENCE/4), verbose=1),
            EarlyStopping(patience = PATIENCE)
            ]

train_gen = train_data_gen.flow(
    X_train, 
    Y_train,
    batch_size = 128
)

model = models.build_dnn(dropout = 0.2)

model.fit_generator(
    train_gen,
    # steps_per_epoch = X_train.shape[0],
    samples_per_epoch = X_train.shape[0],    
    epochs = 1000,
    # epochs = 5, # test 用    
    validation_data = (X_val, Y_val),
    # validation_steps = 256,
    # callbacks = [early_stopping]
    callbacks=callback
)
# model.fit(X_train, Y_train, batch_size = 512, epochs = 200)
rw.save_model(model)

# read test set
X_test = rw.read_dataset('data/test.csv', labeled_data = False)


# result = model.predict_classes(X_test)
result = model.predict(X_test)
result = utils.to_nb_class(result)

rw.save(data = result)
