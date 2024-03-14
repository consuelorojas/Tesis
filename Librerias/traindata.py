import sys
sys.path.append('./data')
sys.path.append('./Librerias')

import utils
import caract as dc
import dataset as ds
import pandas as pd


'''
Creates the train, validation and test data for the model. Does not apply any transformation
neither creates the different sequences used for the model.

'''

class trainData():

    def __init__(self, fpath, fname, column):
        self.fpath = fpath
        self.fname = fname
        self.column = column
        self.data = self.load_data()
        self.train = self.split_data(self.column)[0]
        self.test = self.split_data(self.column)[1]
        self.val = self.split_data(self.column)[2]

    def load_data(self, cutoff = [8/1000, 11/1000]):

        x = ds.MatFileToDataFrame(self.fpath, self.fname)
        y = x.get_dataframe(cutoff)

        x = dc.CaractDefect(y)
        t = x.get_tau()[1]

        h, _ = x.get_hilbert()

        data = pd.merge(y, h, on = 'Hilbert Transform', how = 'outer')
        
        return data
    

    def split_data(self, column, ratio = 0.7):
        
        x = self.data[column].values
        train, val = utils.split_data(x, ratio)
        val, test = utils.split_data(val, 0.5)

        return train, val, test
    
    def get_train(self):
        return self.train
    
    def get_test(self):
        return self.test
    
    def get_val(self):
        return self.val
    