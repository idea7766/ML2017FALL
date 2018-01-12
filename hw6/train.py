import os
import numpy as np
from keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint, TensorBoard, ReduceLROnPlateau

import utils
import models

EPOCH = 1000
VALI = 0.1
PATIENCE = 100
BATCH = 256

ENCODER = 'dnn'
MONITOR = 'val_loss'
# MONITOR = 'loss'


TRAIN_DATA = './data/image.npy'

TRAIN_FOLDER = './model_chk/auto_en_'+ ENCODER + utils.get_time_mark() 
CHECK_POINT_NAME = TRAIN_FOLDER + '/model.{epoch:04d}-{' + MONITOR + ':.4f}.h5'
LOG_NAME = TRAIN_FOLDER + '/log.csv'

if not os.path.exists(TRAIN_FOLDER):
    os.makedirs(TRAIN_FOLDER)

train = np.load(TRAIN_DATA) / 255.
train = utils.shuffle(train)
if ENCODER == 'cnn':
    train = train.reshape(-1, 28, 28, 1)
    model = models.build_cnn()
else:
    model = models.build_dnn()

model.summary()


callback = [
            TensorBoard(TRAIN_FOLDER),
            CSVLogger(LOG_NAME, append=True),
            ReduceLROnPlateau(MONITOR, factor=0.1, patience=int(PATIENCE/2), verbose=1),
            EarlyStopping(MONITOR, patience = PATIENCE),
            ModelCheckpoint(CHECK_POINT_NAME, monitor=MONITOR, period=1)
            ]


model.fit(train, train,
          batch_size = BATCH, 
          epochs = EPOCH, 
          validation_split=VALI,
          callbacks = callback)