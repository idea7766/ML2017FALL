import argparse
from keras.utils.vis_utils import plot_model
from keras.models import load_model

model = load_model('./model_re_bst.h5')
model.summary()
plot_model(model,to_file='./model.png')