import os
import sys

import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from keras.models import Sequential

import read_n_write as rw
import utils
import models

base_dir = os.path.dirname(__file__)
DATA_PATH = sys.argv[1]
LOG_FOLDER = './model_log'
CHECK_POINT_NAME = LOG_FOLDER + '/model.{epoch:04d}-{val_acc:.4f}.h5'
LOG_NAME = LOG_FOLDER + '/log.csv'

PATIENCE = 50
EPOCH = 2000

feats, lables, _ = rw.read_dataset(DATA_PATH, shuffle = True)
X_train, Y_train, X_val, Y_val = utils.validation(feats, lables, 0.05)

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

model = models.build_model(dropout_conv = 0.2, dropout_fully = 0.2)

model.fit_generator(
    train_gen,
    samples_per_epoch = X_train.shape[0],    
    epochs = EPOCH,
    validation_data = (X_val, Y_val),
    callbacks=callback
)
rw.save_model(model)
