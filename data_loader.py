import numpy as np
from os.path import join

class Loader:
    X_train = None
    y_train = None
    X_valid = None
    y_valid = None
    X_test = None
    y_test = None

    X_seq_train = None
    y_seq_train = None
    X_seq_valid = None
    y_seq_valid = None
    X_seq_test = None
    y_seq_test = None

    def load_pretrain(self, fold=0):
        path = './20_fold_data'
        npz = np.load(join(path,str(fold)+'.npz'))
        self.X_train = npz['X_train']
        self.y_train = npz['y_train']
        self.X_valid = npz['X_valid']
        self.y_valid = npz['y_valid']
        self.X_test = npz['X_test']
        self.y_test = npz['y_test']

    def load_finetune(self, fold=0 ):
        path = './20_fold_data'
        npz = np.load(join(path,str(fold)+'.npz'))
        self.X_seq_train = npz['X_seq_train']
        self.y_seq_train = npz['y_seq_train']
        self.X_seq_valid = npz['X_seq_valid']
        self.y_seq_valid = npz['y_seq_valid']
        self.X_seq_test = npz['X_seq_test']
        self.y_seq_test = npz['y_seq_test']

if __name__ == '__main__':
    d = Loader()
    d.load_pretrain()
    print(d.X_train.shape)