import os
import random 
import time
import numpy as np
import pandas as pd
from keras.models import load_model
from keras.utils.np_utils import to_categorical
base_dir = os.path.dirname(__file__)

def read_dataset(path, labeled_data = True, shuffle = False):
    '''
    # Arguments
    path : 相對位址 (relative address)
    labeled_data : 表示 x 是 data
        True : 表示是要處理 trainig set\n
        False : 表示是要處理 testinig set
    shuffle : 決定是否打亂 datas ,但 testing set 無法打亂
    # Return
    traning set : 2-D array, label, line_index (共3項)\n
    testing set : 2-D array
    '''
    print('\nReading File...')

    data_dir = os.path.join(base_dir, path)
    with open(data_dir) as file:
        print('--Open file:',data_dir)
        if labeled_data:
            sta = 'Traning Set'
        else:
            sta = 'Testing Set'
        print('--Stament :',sta )

        datas = []

        for ind, line in enumerate(file):

            if labeled_data:
                lable, feat = line.split(',')
            else:
                _, feat = line.split(',')

            feat = np.fromstring(feat, dtype = int, sep = ' ')
            try:
                feat = np.reshape(feat, (48, 48, 1))
            except:
                print('--Exception at line', ind)
                continue
            
            if labeled_data:
                datas.append((feat, int(lable), ind))
            else:
                datas.append(feat)
        if shuffle and labeled_data:
            random.shuffle(datas)
            
        if labeled_data:
            feats, lables, line_ind = zip(*datas) #這裡因為 datas 儲存了多變數
        else:
            feats = datas

        feats = np.asarray(feats)
        print('--feats shape :', np.shape(feats))

        if labeled_data:
            lables = to_categorical(np.asarray(lables, dtype = float))
            print('--lables shape :', np.shape(lables))
            return feats / 255, lables, line_ind
        else:
            return feats / 255


def save(data = None, name = 'ans', extension = 'csv', time_mark = True, path = None):
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
    if path == None:
        path = base_dir + '/ans/' + save_name
    ans = []
    for i in range(len(data)):
        ans.append((i, data[i]))

    df_ans = pd.DataFrame(ans, index = None, columns = ['id', 'label'])
    df_ans.to_csv(path, index = False)
        
    print('--Save as :', path)

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
        ans.append((i, data[i]))

    df_ans = pd.DataFrame(ans, index = None, columns = ['id', 'label'])
    df_ans.to_csv(path, index = False)
        
    print('--Save as :', path)

def save_model(model = None, name = 'model', extension = 'h5', time_mark = True):
    print('\nSaving model...')
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

    save_path = base_dir + '/model/' + save_name
    model.save(save_path)
    print('--Save as :', save_path)

