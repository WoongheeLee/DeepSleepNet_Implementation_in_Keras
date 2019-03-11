import os
from os import listdir
from os.path import isfile, join, splitext
import numpy as np
import random

seed = 0

np.random.seed(seed)

class Data_loader:
    X_seq = None
    y_seq = None
    X_seq_train = None
    y_seq_train = None
    X_seq_valid = None
    y_seq_valid = None
    X_seq_test = None
    y_seq_test = None

    X_train = None
    y_train = None
    X_valid = None
    y_valid = None
    X_test = None
    y_test = None

    def data_preparation(self, mypath="./data/eeg_fpz_cz"):
        # get file list
        file_list = [f for f in listdir(mypath) if isfile(join(mypath, f))]

        data_X, data_y = [], []
        for i in range(len(file_list)):
            with np.load(join(mypath,file_list[i])) as npz:
                data_X.append(npz['x'])
                data_y.append(npz['y'])

        # one-hot encoding sleep stages
        temp_y = []
        for i in range(len(data_y)):
            temp_ = []
            for j in range(len(data_y[i])):
                temp = np.zeros((5,))
                temp[data_y[i][j]] = 1.
                temp_.append(temp)
            temp_y.append(np.array(temp_))
        data_y = temp_y

        # make sequence data
        seq_length = 25 # 원 저자는 시퀀스 길이를 25로 함 (30초 * 25개 시퀀스)

        X_seq, y_seq = [], []

        for i in range(len(data_X)):
            for j in range(0, len(data_X[i]), seq_length): # discard last short sequence
                if j+seq_length < len(data_X[i]):
                    X_seq.append(np.array(data_X[i][j:j+seq_length]))
                    y_seq.append(np.array(data_y[i][j:j+seq_length]))
        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq)

        self.X_seq = X_seq
        self.y_seq = y_seq

    def rotate(self, l, k):
        n = int(len(l)*1/20*k)
        l = l[-n:] + l[:-n]
        return l

    def get_k_th_seq(self, X_seq, y_seq, k):
        seq_idx = [i for i in range(len(X_seq))]
        random.shuffle(seq_idx)
        seq_idx = self.rotate(seq_idx, k)
         
        idx_train = int(len(X_seq)*0.8)
        idx_valid = int(len(X_seq)*0.1)+1
        idx_test = int(len(X_seq)*0.1)+1

        X_seq_train, y_seq_train = [], []
        X_seq_valid, y_seq_valid = [], []
        X_seq_test, y_seq_test = [], []

        for i in range(0, idx_train):
            idx = seq_idx[i]
            X_seq_train.append(X_seq[idx])
            y_seq_train.append(y_seq[idx])

        for i in range(idx_train, idx_train+idx_valid):
            idx = seq_idx[i]
            X_seq_valid.append(X_seq[idx])
            y_seq_valid.append(y_seq[idx])

        for i in range(idx_train+idx_valid, len(seq_idx)):
            idx = seq_idx[i]
            X_seq_test.append(X_seq[idx])
            y_seq_test.append(y_seq[idx])

        X_seq_train = np.array(X_seq_train)
        y_seq_train = np.array(y_seq_train)

        X_seq_valid = np.array(X_seq_valid)
        y_seq_valid = np.array(y_seq_valid)

        X_seq_test = np.array(X_seq_test)
        y_seq_test = np.array(y_seq_test)

        
        self.X_seq_train = X_seq_train
        self.y_seq_train = y_seq_train
        self.X_seq_valid = X_seq_valid
        self.y_seq_valid = y_seq_valid
        self.X_seq_test = X_seq_test
        self.y_seq_test = y_seq_test

    # This method should follow right after get_k_th_seq
    def get_k_th_data(self,X_seq_train, y_seq_train, X_seq_valid, y_seq_valid, X_seq_test, y_seq_test):
        X_train, y_train = [], []
        X_valid, y_valid = [], []
        X_test, y_test = [], []

        for i in range(len(X_seq_train)):
            for j in range(len(X_seq_train[i])):
                X_train.append(X_seq_train[i][j])
                y_train.append(y_seq_train[i][j])
                    
        for i in range(len(X_seq_valid)):
            for j in range(len(X_seq_valid[i])):
                X_valid.append(X_seq_valid[i][j])
                y_valid.append(y_seq_valid[i][j])

        for i in range(len(X_seq_test)):
            for j in range(len(X_seq_test[i])):
                X_test.append(X_seq_test[i][j])
                y_test.append(y_seq_test[i][j])

        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)
        self.X_valid = np.array(X_valid)
        self.y_valid = np.array(y_valid)
        self.X_test = np.array(X_test)
        self.y_test = np.array(y_test)

if __name__=="__main__":
    d = Data_loader() 
    d.data_preparation()

    path = './20_fold_data'
    if os.path.exists(path) is False:
        os.mkdir(path)
    
    for i in range(20):
        d.get_k_th_seq(d.X_seq, d.y_seq, i)
        d.get_k_th_data(d.X_seq_train, d.y_seq_train, d.X_seq_valid, d.y_seq_valid, d.X_seq_test, d.y_seq_test) 
        with open(join(path,str(i)+'.npz'),'wb') as f:
            np.savez(
                f, 
                X_seq_train = d.X_seq_train,
                y_seq_train = d.y_seq_train,
                X_seq_valid = d.X_seq_valid,
                y_seq_valid = d.y_seq_valid,
                X_seq_test = d.X_seq_test,
                y_seq_test = d.y_seq_test,
                X_train = d.X_train,
                y_train = d.y_train,
                X_valid = d.X_valid,
                y_valid = d.y_valid,
                X_test = d.X_test,
                y_test = d.y_test
            )
        print(i)

    print("done")


