import os
import numpy as np
from keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint, TensorBoard, ReduceLROnPlateau

import utils
import models

EPOCH = 1000
NUM_GRAB_VAL = 10
VALI = 0.1
DROP = 0.1
PATIENCE = 8
BATCH = 256
MONITOR = 'val_rmse'

NUM_USR = 6041
NUM_MOV = 3953

MAGIC_MEAN = 3.58171208604

TRAIN_DATA = './data/train.csv'
DATA_DIR = './data/'
COUNTER_PATH = './counter/mf_count'

NUM_FOLDER = utils.load_pkl(COUNTER_PATH)
NUM_FOLDER += 1
utils.save_pkl(COUNTER_PATH, NUM_FOLDER)

TRAIN_FOLDER = './model_chk/mf_' + str(NUM_FOLDER) + '_drop_' + str(DROP)
CHECK_POINT_NAME = TRAIN_FOLDER + '/model.{epoch:04d}-{' + MONITOR + ':.4f}.h5'
LOG_NAME = TRAIN_FOLDER + '/log.csv'

if not os.path.exists(TRAIN_FOLDER):
    os.makedirs(TRAIN_FOLDER)

str_grab = str(NUM_GRAB_VAL)
x_train ,y_train = utils.load_data(TRAIN_DATA, file_type = 'train')
print(y_train)
y_train = y_train.astype(float) - MAGIC_MEAN
print(y_train)

x_train, y_train = utils.shuffle(x_train, y_train)


callback = [
            TensorBoard(TRAIN_FOLDER),
            CSVLogger(LOG_NAME, append=True),
            ModelCheckpoint(CHECK_POINT_NAME, monitor=MONITOR, period=1),
            ReduceLROnPlateau(MONITOR, factor=0.1, patience=int(PATIENCE/4), verbose=1),
            EarlyStopping(MONITOR, patience = PATIENCE)
            ]

model = models.build_mf(num_usr = NUM_USR, num_movie = NUM_MOV)

model.fit(np.hsplit(x_train, 2), y_train,
          batch_size = BATCH, 
          epochs = EPOCH, 
          validation_split=VALI,
        #   validation_data = (np.hsplit(x_val, 2), y_val),
          callbacks = callback)
