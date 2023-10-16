import scipy.io as io

def load_data(fpath, fname):
    with open(f'{fpath}/{fname}', 'rb') as f:
        data= io.loadmat(f)
    return data