from keras.models import load_model, Sequential
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
import itertools
import numpy as np
import matplotlib.pyplot as plt

import utils
import read_n_write as rw

MODEL_PATH = './model_re_bst.h5'
DATA_PATH = './data/train.csv'

model = load_model(MODEL_PATH)
x, y, _ = rw.read_dataset(DATA_PATH)
y = utils.to_nb_class(y)

y_pred = model.predict_classes(x)
# y_pred = to_categorical(y_pred)
# y_pred = utils.to_nb_class(y_pred)
print(y_pred)

cnf_mat = confusion_matrix(y, y_pred)


def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.jet):
    """
    This function prints and plots the confusion matrix.
    """
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, '{:.2f}'.format(cm[i, j]), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

plt.figure()
plot_confusion_matrix(cnf_mat, classes=["Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral"])
plt.show()