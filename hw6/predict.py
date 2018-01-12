import sys
import numpy as np

import utils

# TEST_PATH = './data/test_case.csv'
TEST_PATH = sys.argv[1]
OUT_PUT_PATH = sys.argv[2]
LABEL_PATH = './label.npy'

# SAVE_ANS = 

label = np.load(LABEL_PATH)
test_case = utils.load_test(TEST_PATH)

ans = np.zeros(len(test_case))
for i in range(len(test_case)):
    if label[test_case[i, 0]] == label[test_case[i, 1]]:
        ans[i] = 1

print(ans)
print(ans.shape)
utils.save_dir(ans, OUT_PUT_PATH)