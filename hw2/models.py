import numpy as np
from numpy.linalg import inv

import utils

def logistic_regression(x, y, lr = 0.1, epoch = 10000, batch = 32, optimizer = 'static', shuffle = True, validation_rate = 0., early_stopping = False, patience = 0):
    num_data = x.shape[0]
    num_fea = x.shape[1]

    w = np.zeros(num_fea + 1)
    x = np.insert(x, 0, 1, axis = 1)

    sum_squ_grad = 0
    pre_acc = None
    pre_w = None
    best_acc = 0

    if shuffle == True:
        x, y = utils.shuffle(x, y)

    x_train, y_train, x_val, y_val = utils.validation(x, y, ratio = validation_rate)
    
    for i in range(epoch):
        batch_ind = i * batch % num_data
        batch_end = (batch_ind + batch) % num_data
        x_bat = x_train[batch_ind : batch_end]
        y_bat = y_train[batch_ind : batch_end]
        grad = gradient(x, y, w)

        if optimizer == 'adagrad':
            ada, sum_squ_grad = adagrad(grad, sum_squ_grad)
            w = w - (lr / ada) * grad
        elif optimizer == 'static':
            w = w - lr * grad
        
        y_pred = predict(x_val[: , 1:], w[0], w[1:])
        
        val_acc = acc(y_val, y_pred)
        print('>>> epoch : %d \t  | val acc: %f' %(i + 1, val_acc)) # 改
        if best_acc <= val_acc:
            best_acc = val_acc
            best_w = w
        if early_stopping == True and (i+1) % 100 == 0 :
            if pre_acc == None:
                pre_acc = val_acc
            else:
                if pre_acc > val_acc and patience == 0:
                    break
                elif pre_acc > val_acc:
                    patience -= 1
                else:
                    pre_acc = val_acc
    return best_w[0], best_w[1:]

def gaussian_distribution(x, y):
    num_data = x.shape[0]
    num_fea = x.shape[1]
    # find mean (mu)
    mu0 = np.zeros(num_fea)
    mu1 = np.zeros(num_fea)
    # print('mu0',mu0)
    # print('mu1',mu1)        

    count0, count1 = 0, 0

    for i, lab in zip(range(num_data), y):
        if lab == 1:
            mu1 += x[i]
            count1 += 1
        else:
            mu0 += x[i]
            count0 += 1
    mu1 = mu1 / count1
    mu1 = np.matrix(mu1).transpose()
    mu0 = mu0 / count0
    mu0 = np.matrix(mu0).transpose()    
    # print('mu0',mu0)
    # print('mu0 shape',mu0.shape)
    # print('mu1',mu1)    
    # find covariance (sigma)    
    sigma0 = np.matrix(np.zeros((num_fea, num_fea)))
    sigma1 = np.matrix(np.zeros((num_fea, num_fea)))
    
    # 從這開始轉換成正規矩陣運算
    for i, lab, fea in zip(range(num_data), y, x):
        x_mat = np.matrix(fea).transpose()
        if lab == 1:
            diff = x_mat - mu1
            sigma1 += diff * diff.transpose()
        else:
            diff = x_mat - mu0
            sigma0 += diff * diff.transpose()
    sigma = sigma1 / count1 + sigma0 / count0
    # print('sigma', sigma)
    # print('sigma0', sigma0)
    # print('sigma1', sigma1)    
    

    sigma_inv = inv(sigma)
    w = (mu1 - mu0).transpose() * sigma_inv
    b = -0.5 * mu1.transpose() * sigma_inv * mu1 +\
        0.5 * mu0.transpose() * sigma_inv * mu0 +\
        np.log(count1 / count0)
    w = np.array(w)
    return float(b), w.flatten()
        
def sigmoid(z):
    sig = 1 / (1 + np.exp(-z))
    return np.clip(sig, 1e-8, 1-(1e-8))

def gradient(x, y, w):
    '''
    x: [1, x1, x2, ...]
    w: [bias, w1, w2, ...]
    '''
    num_data = x.shape[0]
    num_fea = x.shape[1]
    gradient_w = np.zeros(num_fea)

    z = np.dot(x, w)
    hypothesis = sigmoid(z)

    loss = hypothesis - y    
    
    gradient_w = np.dot(np.transpose(x), loss) / num_data

    return gradient_w

def acc(y, y_pred):
    num_data = y.shape[0]
    diff = y - y_pred
    diff = np.absolute(diff)
    num_diff = int(np.sum(diff))
    rate = 1 - (num_diff / num_data)
    return rate

def predict(x, b, w):
    z = np.dot(x, w) + b
    y = np.around(sigmoid(z))
    return y.flatten()

def adagrad(gradient, sum_squ_grad):
    sum_squ_grad += gradient ** 2
    adagrad = sum_squ_grad ** 0.5
    return adagrad, sum_squ_grad 
