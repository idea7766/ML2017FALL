import sys
import time
import re
import pandas as pd
import numpy as np
from keras.preprocessing import text


###################################
# -----------NEW UTILS----------- #
###################################
def load_data(path, file_type = 'train'):
    '''
    ## Attribute
    path: data path
    type: 3 formats of data
        train: label  +++$+++ sentence\n
        train_nolabel: setence\n
        test: id, setence\n
    '''  
    print('\nReading:', path)
    print('** if exception happen, it should be handel. DONT force terminate.')
    if file_type == 'train':
        with open(path) as f:
            label = []
            setence = []
            for line in f:
                content = line.split(' +++$+++ ')
                label.append(content[0])
                setence.append(content[1])
            label = np.array(label, dtype = np.int)
            return label, setence
    elif file_type == 'train_nolabel':
        with open(path) as f:
            setence = []
            for line in f:
                setence.append(line)
            return setence
    elif file_type == 'test':
        with open(path) as f:
            index = np.array([])
            setence = []
            for i, line in enumerate(f):
                content = line.split(',', 1)
                # print(content[0])
                # print(content[1])
                try:
                    index = np.append(index, int(content[0]))
                    setence.append(content[1])
                except:
                    print('--Exception at line', i, 'type:', sys.exc_info())
            # index = np.array(index, dtype = np.int)
            # return index, setence
            return setence
    else:
        pass

def split_setence(setence):
    setenece_split = []
    for line in setence:
        words = text.text_to_word_sequence(line, filters='!"\'#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
        setenece_split.append(words)
    return setenece_split

def split_setence_char(setence):
    setenece_split = []
    for line in setence:
        words = re.split(r'(\W|\s)\s*',  line.lower())
        setenece_split.append(words)
    return setenece_split

def w2id(setence, vocab):
    id_setence = []
    for line in setence:
        id_line = []
        for word in line:
            id = vocab.get(word)
            if id is None:
                id = 0
            id_line.append(id)
        id_setence.append(id_line)
    return id_setence

# Wrong
def vec_setence_gen(setence, label, model):
    for line, y in zip(setence, label):
        vec = []
        for word in line:
            try:
                vec.append(model[word])
            except:
                vec.append(np.zeros(100))
        yield (np.array(vec), y)

# word matrix
def get_embedding_matrix(wv_model):
    return wv_model.wv.syn0


###################################
# -----------OLD UTILS----------- #
###################################

def validation(feats, lables, ratio = 0.1):
    '''
    # Arguments
    reatio : 多少比例的資料要被當作 validation set 
    # Return
    X_train, Y_train, X_val, Y_val
    '''
    nb_cut = int(np.shape(feats)[0] *ratio)

    X_train = feats[:-nb_cut]
    Y_train = lables[:-nb_cut]

    X_val = feats[-nb_cut:]
    Y_val = lables[-nb_cut:]

    return X_train, Y_train, X_val, Y_val

def to_nb_class(lables):
    nb_lables = []
    length = len(lables)
    for col in lables:
        nb_class = np.argmax(col)
        nb_lables.append(nb_class)

    nb_lables = np.asarray(nb_lables)

    return nb_lables

def save_ans(data = None, folder_path = './ans', name = 'ans', extension = 'csv', time_mark = True):
    '''
    # 儲存成csv 用
    ## Arguments
    data : 欲儲存的參數
    name : 將檔案儲存在 ans 資料夾底下，名稱為 name 
    extension : 副檔名
    time_mark
    '''
    print('\nSaving file...')
    if time_mark:
        save_name = name + '_' + \
        str(time.localtime().tm_year)[-2:] + \
        str(time.localtime().tm_mon).zfill(2) + \
        str(time.localtime().tm_mday).zfill(2) + '_' + \
        str(time.localtime().tm_hour).zfill(2) + '-' + \
        str(time.localtime().tm_min).zfill(2) +'-' + \
        str(time.localtime().tm_sec).zfill(2) + '.' +\
        extension

    else:
        save_name = name
    
    save_path = folder_path + '/' + save_name
    ans = []
    for i in range(len(data)):
        ans.append((i, int(data[i])))

    df_ans = pd.DataFrame(ans, index = None, columns = ['id', 'label'])
    df_ans.to_csv(save_path, index = False)
        
    print('--Save as :', save_path)

def save_dir(data = None, path = None):
    '''
    # 用路徑儲存
    ## Arguments
    data : 欲儲存的參數
    path : PATH
    '''
    print('\nSaving file...')
    ans = []
    for i in range(len(data)):
        ans.append((i, int(data[i])))

    df_ans = pd.DataFrame(ans, index = None, columns = ['id', 'label'])
    df_ans.to_csv(path, index = False)
        
    print('--Save as :', path)