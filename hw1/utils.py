import numpy as np
import pandas as pd

def rescaling(data):
    '''
    Arrange all feature btw [0, 1]
    '''
    if data.ndim > 2:
        raise('不支援 2 維以上的 array 喔喔喔喔喔喔喔')
    else:
        max, min = np.amax(data, axis = 0), np.amin(data, axis = 0)
        new_data = (data - min) / (max - min)
    return new_data, max, min

def scaling(data, max, min):
    '''
    Arrange all feature btw [0, 1]
    '''
    if data.ndim > 2:
        raise('不支援 2 維以上的 array 喔喔喔喔喔喔喔')
    else:
        new_data = (data - min) / (max - min)
    return new_data

def load(path, mode = 'train', fea_select = None, y_pos = 0): # only for ML2017FALL hw1
    '''
    # HW1 的 data loader
    ## Attribute
    path: 要讀取的檔案路徑
    mode: 有 'train' 及 'test' 對應要讀的檔案類型
    fea_select: 要的特徵
    y_pos: 經過讀取後，y 的位置
    ## Return
    b: bias
    w: weight
    '''
    if mode == 'train':
        df_pm25 = pd.read_csv(path, encoding='big5')
        #將 'NR' 改成 0
        arr_pm25_pre = df_pm25.iloc[:,3:].replace('NR', 0).values
        arr_pm25_pre = arr_pm25_pre.astype(float)

        arr_pm25 = np.array([arr_pm25_pre[:18, 0]])

        day_num = int(arr_pm25_pre.shape[0] / 18)
        hour_of_day = 24
        for i in range(day_num):
            for j in range(hour_of_day):
                if i==0 and j == 0:
                    continue
                data_of_hour = arr_pm25_pre[18*i:18+18*i, j]
                arr_pm25 = np.append(arr_pm25, [data_of_hour], axis = 0)
        # 加入風的 x, y 向量
        wind_x = np.cos(arr_pm25[:, 15] * np.pi / 180).reshape(arr_pm25.shape[0],1)
        wind_y = np.sin(arr_pm25[:, 15] * np.pi / 180).reshape(arr_pm25.shape[0],1)        
        arr_pm25 = np.append(arr_pm25, wind_x, axis = 1)
        arr_pm25 = np.append(arr_pm25, wind_y, axis = 1)   
        print('arr_pm25.shape', arr_pm25.shape)
        if fea_select != None:
            arr_pm25 = arr_pm25[:, fea_select]

        data_of_month = int(arr_pm25.shape[0] / 12)
        arr_pm25_new = align(arr_pm25[0: data_of_month], 9)
        x_new = arr_pm25_new [:-1]
        y_new = arr_pm25_new [1:, y_pos]
        for i in range(12):
            if i == 0:
                continue
            arr_pm25_temp = align(arr_pm25[data_of_month * i : data_of_month * (i + 1)], 9)
            x_new = np.append(x_new, arr_pm25_temp[:-1], axis = 0)
            y_new = np.append(y_new, arr_pm25_temp[1:, y_pos], axis = 0)

        return x_new, y_new.flatten()

    if mode == 'test':
        df_pm25 = pd.read_csv(path, encoding='big5', header=None)
        #將 'NR' 改成 0
        arr_pm25_pre = df_pm25.iloc[:,2:].replace('NR', 0).values
        arr_pm25_pre = arr_pm25_pre.astype(float)
        arr_pm25 = np.array(np.transpose(arr_pm25_pre[:18]))

        wind_x = np.cos(arr_pm25[:, 15] * np.pi / 180).reshape(arr_pm25.shape[0],1)
        wind_y = np.sin(arr_pm25[:, 15] * np.pi / 180).reshape(arr_pm25.shape[0],1)        
        arr_pm25 = np.append(arr_pm25, wind_x, axis = 1)
        arr_pm25 = np.append(arr_pm25, wind_y, axis = 1)

        if fea_select != None:
            arr_pm25 = arr_pm25[:, fea_select]
        arr_pm25 = [arr_pm25.flatten()]
        data_num = int(arr_pm25_pre.shape[0]/18)
        for i in range(data_num):
            if i == 0:
                continue
            data_of_9hr = np.transpose(arr_pm25_pre[18*i:18*i+18])
            wind_x = np.cos(data_of_9hr[:, 15] * np.pi / 180).reshape(data_of_9hr.shape[0],1)
            wind_y = np.sin(data_of_9hr[:, 15] * np.pi / 180).reshape(data_of_9hr.shape[0],1)        
            data_of_9hr = np.append(data_of_9hr, wind_x, axis = 1)
            data_of_9hr = np.append(data_of_9hr, wind_y, axis = 1) 
            if fea_select != None:
                data_of_9hr = data_of_9hr[:, fea_select]
            arr_pm25 = np.append(arr_pm25, [data_of_9hr.flatten()], axis=0)

        return arr_pm25


def align(data, num):
    if num <= 0 or None:
        raise('請輸入大於 0 的數值')

    run = data.shape[0] - num + 1 # 毎 n 比當成 1 筆 data
    ls_align_data = np.array([data[0:num].flatten()])
    for ind in range(run):
        if ind == 0:
            continue
        ls_align_data = np.append(ls_align_data, [data[ind:ind+num].flatten()], axis = 0)
    new_data = np.asarray(ls_align_data)
    return new_data

def save_ans(data, path):
    col = ['id', 'value']
    ans_sheet = []
    for i in range(data.shape[0]):
        if data[i] < 0:
            data[i] = 0
        ans_sheet.append(('id_' + str(i), data[i]))
    df_ans = pd.DataFrame(ans_sheet, index = None, columns = col)
    df_ans.to_csv(path, index=False)
    print(">>> ans儲存成功")

def save_model(b, w, path):
    if w.ndim != 1:
        raise('不支援 ndim != 1 的arr')
    b_w = np.insert(w, 0, b) # bias 放第 0 項
    np.save(path, b_w)
    print(">>> model 儲存存成功")
    
def save_scaler(max, min, path):
    scaler = np.append([max], [min], axis = 0)
    np.save(path, scaler)
    print('>>> scaler 儲存成功')    

def load_model(path):
    w = np.load(path)
    return w[0], w[1:]

def load_scaler(path):
    scaler = np.load(path)
    return scaler[0], scaler[1]

def validation(feats, lables, ratio = 0.1):
    '''
    # Arguments
    ratio : 多少比例的資料要被當作 validation set 
    # Return
    X_train: , Y_train: , X_val: , Y_val:
    '''
    nb_cut = int(feats.shape[0] *ratio)

    x_train = feats[:-nb_cut]
    y_train = lables[:-nb_cut]

    x_val = feats[-nb_cut:]
    y_val = lables[-nb_cut:]

    return x_train, y_train, x_val, y_val

def shuffle(x, y):
    '''
    # shuffle x 和 y
    '''
    fea_num = x.shape[1]
    data = np.insert(x, fea_num, y, axis =1)
    np.random.shuffle(data)
    x = data[:, :fea_num]
    y = data[:, fea_num:]
    return x, y.flatten()
