import os
import argparse
import numpy as np
from skimage import io, transform

# import plot_utils as pu
import utils

IMG_FOLDER = './data/Aberdeen'
SVD_PATH = './svd.pkl'
U4_PATH = './u4.pkl'
SIZE = 100

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--q1', action = 'store_true')
    parser.add_argument('--q2', action = 'store_true')
    parser.add_argument('--q3', action = 'store_true')
    parser.add_argument('--q4', action = 'store_true')
    parser.add_argument('-g', action = 'store_true', help='bounding@[0, 1] q3\'s faces')    
    parser.add_argument('-save', action = 'store_true', help='save svd info')        
    parser.add_argument('-s', type = int, default=100, help='get size')
    parser.add_argument('-k', type = int, default=4, help='k dim')    
    
    args = parser.parse_args()

    print('PCA Log')    
    # global
    print('- Get x ndarray')
    size = args.s
    print('  * size:', size)
    k = args.k
    x = get_data_arr(size)
    print('  * x shape:', x.shape)
    x_mean = np.mean(x)
    print('  * x maen:', x_mean)
    # SVD
    U, s, V = np.linalg.svd(x.T - x_mean, full_matrices=False)
    # print(U.T)
    eig_face = U.T

    if args.save:
        utils.save_pkl(SVD_PATH, (U, s, V))
        utils.save_pkl(U4_PATH, eig_face[:4])

    if args.q1:
        print('- A1')         
        x_face_mean = np.mean(x, axis=0)
        x_face_mean = x_face_mean.reshape(-1, size, size , 3)
        print('  * face maen shape:', x_face_mean.shape)
        
        # pu.show_eng_face(x_face_mean, row=1, col=1, plot_neg=False, n=1)

    if args.q2:
        print('- A2')                 
        eig_face = eig_face.reshape(-1, size, size , 3)
        print('  * eigen face shape:', eig_face.shape)        
        # pu.show_eng_face(eig_face)

    if args.q2:
        print('- A2')                 
        eig_face = eig_face.reshape(-1, size, size , 3)
        print('  * eigen face shape:', eig_face.shape)        
        # pu.show_eng_face(eig_face)


    if args.q3:
        print('- A3') 
        en = np.matrix(eig_face[:k])
        print('  * encoder shape:', en.shape)
        constuct_matrix = en.T * en
        utils.save_pkl('./construct_matrix.pkl', constuct_matrix)
        print('  * constuct matrix shape:', constuct_matrix.shape)
        sample = get_sample(size=size, sample=(0, 4, 10, 14))
        sampleT = np.matrix(sample.T - x_mean)
        sample = sample.reshape(-1, size, size, 3)
        print('  * sampleT matrix shape:', sampleT.shape)        
        result = constuct_matrix * sampleT
        result = np.array(result.T) + x_mean
        print('  * result shape:', result.shape)                
        faces = result.reshape(-1, size, size, 3)
        faces -= np.min(faces)
        faces /= np.max(faces)
        faces = (faces * 255).astype(np.uint8)          
        print('  * faces shape:', faces.shape)
        # print(faces)       
        # pu.show_recon(sample, faces)        

    if args.q4:
        print('- A4')         
        avg_s = s[:4] / np.sum(s)
        print('  * avg s:', avg_s)

def get_data_arr(size = 100):
    for f_path in os.listdir(IMG_FOLDER):
        img = io.imread(os.path.join(IMG_FOLDER, f_path))
        if size != 600:
            img = transform.resize(img, (size, size, 3), mode ='constant')
        if 'x' in locals():
            x = np.append(x, img.flatten().reshape(1, -1), axis=0)
        else:
            x = img.flatten().reshape(1, -1)
    return x

def get_sample(size=100, sample=None):
    for i in sample:
        f_path = str(i)+'.jpg'
        img = io.imread(os.path.join(IMG_FOLDER, f_path))
        img = transform.resize(img, (size, size, 3), mode ='constant')
        if 'x' in locals():
            x = np.append(x, img.flatten().reshape(1, -1), axis=0)
        else:
            x = img.flatten().reshape(1, -1)
    return x
    

if __name__ == '__main__':
    main()