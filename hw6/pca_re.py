import os
import sys
import argparse
import numpy as np
from skimage import io, transform

import utils

SVD_PATH = './svd.pkl'
U4_PATH = './u4.pkl'
X_MAEN = './x_mean.pkl'

INPUT = sys.argv[1]
OUTPUT = sys.argv[2]

def main():
    size = 600
    k = 4

    # (U, s, V) = utils.load_pkl(SVD_PATH)
    eig_face = utils.load_pkl(U4_PATH)
    x_mean = utils.load_pkl(X_MAEN)
    # eig_face = U.T

    print('- A3')
    # en = eig_face[:k]
    # en = np.matrix(eig_face[:k])
    en = eig_face
    # en = np.matrix(eig_face)
    print('  * encoder shape:', en.shape)
    sample = get_sample()


    sampleT = np.matrix(sample - x_mean).T
    print('  * sampleT matrix shape:', sampleT.shape)        
    constuct_matrix = en * sampleT
    result = en.T * constuct_matrix
    result = np.array(result.T) + x_mean
    print('  * result shape:', result.shape)                
    faces = result.reshape(-1, size, size, 3)
    faces -= np.min(faces)
    faces /= np.max(faces)
    faces = (faces * 255).astype(np.uint8)          
    print('  * faces shape:', faces.shape)
    # print(faces)       
    io.imsave(OUTPUT, faces.reshape(size, size, 3))

def get_sample():
    img = io.imread(os.path.join(INPUT, OUTPUT)).flatten()
    return img

if __name__=='__main__':
    main()
