import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

TRAIN_ACC_PATH = './train_acc.csv'
VAL_ACC_PATH = './val_acc.csv'

# 讀去取成 arr
acc = pd.read_csv(TRAIN_ACC_PATH)
epoch = np.array(acc)[:, 1].astype(int)
print(epoch)
acc = np.array(acc)[:, 2]

val_acc = pd.read_csv(VAL_ACC_PATH)
val_acc = np.array(val_acc)[:, 2]

font ={'size': 16}
plt.ylabel('accuracy', **font)
plt.xlabel('epochs', **font)
plt.plot(epoch, acc, label = 'train')
plt.plot(epoch, val_acc, label = 'validation')
plt.legend(loc='lower right', shadow=True, fontsize='x-large')
plt.show()
