import pandas as pd
import numpy as np
import os

import utils

train_path = os.path.join(os.path.dirname(__file__), "./data/train.csv")
test_path = os.path.join(os.path.dirname(__file__), "./data/test.csv")

train_data = utils.load(train_path, mode = 'train', fea_select = (4, 5, 6, 9))
test_data = utils.load(test_path, mode = 'test', fea_select = (4, 5, 6, 9))

np.random.shuffle(train_data)
print('train data', train_data.shape)
print('test data',test_data.shape)

 