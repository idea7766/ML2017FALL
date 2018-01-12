from sklearn.cluster import KMeans
from keras.models import load_model
import numpy as np

import models
import utils

SAVE_W = False
ENCODER = 'dnn'

TRAIN_DATA = './data/image.npy'
MODEL = './model.h5'

WEIGHT_PATH = './model_w.h5'
LABEL_PATH = './label.npy'

model = load_model(MODEL)
if SAVE_W:
    model.save_weights(WEIGHT_PATH)

train = np.load(TRAIN_DATA) / 255.

if ENCODER == 'cnn':
    train = train.reshape(-1, 28, 28, 1)
    encoder = models.cnn_encoder(WEIGHT_PATH)
else:
    encoder = models.dnn_encoder(load_model(MODEL))


x = encoder.predict(train)
kmeans = KMeans(n_clusters = 2, verbose=1).fit(x)

np.save(LABEL_PATH, kmeans.labels_)
# print(kmeans.labels_[:100])