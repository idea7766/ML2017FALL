import os
import sys
from keras.models import Sequential, load_model

import utils
import read_n_write as rw

TEST_PATH = sys.argv[1]
ANS_PATH = sys.argv[2]
MODELS_PATH = ['./model_re_bst.h5',
               './model_ta.h5',
               './model_x.h5']
WEIGHT = [1.06, 1, 1]
x_test = rw.read_dataset('data/test.csv', labeled_data = False)

model_ls = []
result_ls = []

num_model = len(MODELS_PATH)

for i, w, mod_path in zip(range(num_model), WEIGHT, MODELS_PATH):
# io processing
    model  = load_model(mod_path)
    model_ls.append(model)
    result = model.predict(x_test)
    result_ls.append(result)
    if i == 0:
        result_total = w * result
    else:
        result_total += w * result
    # print(result_total)
predict = utils.to_nb_class(result_total)
print(predict)
rw.save_dir(predict, path = ANS_PATH)
# print(result_ls[0].shape)
# rw.save(result)

