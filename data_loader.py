import torch.utils.data as tud
import pandas as pd
import torch
from utils import arr2hot

class DataSet(tud.Dataset):
    def __init__(self, data, n_know):
        super(DataSet, self).__init__()
        self.data = data
        self.n_know = n_know

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return int(self.data.iloc[item]['sid']), int(self.data.iloc[item]['pid']), torch.Tensor(arr2hot(self.data.iloc[item]['Q'], self.n_know)), \
               self.data.iloc[item]['rt'], self.data.iloc[item]['r']

class DataLoader(object):
    def __init__(self, dataset):
        print("loading data...")
        self.dataset = dataset
        if dataset == "junyi":
            self.n_know, self.n_stu, self.n_exer = 39, 15000, 718
            self.data_group = [0, 6, 10, 18, 26, 35, 39]
        elif dataset == "assistment2017":
            self.n_know, self.n_stu, self.n_exer = 102, 1709, 3162
            self.data_group = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 102]
        elif dataset == "ednet":
            self.n_know, self.n_stu, self.n_exer = 188, 9980, 12165
            self.data_group = [0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 188]
        elif dataset == "pisa2015":
            self.n_know, self.n_stu, self.n_exer = 11, 1476, 17
            self.data_group = [0, 5, 9, 11]

    def load_test_dataSet(self, batch_size):
        print('loading test data...')
        test_data = pd.read_pickle('./data/test/test_' + self.dataset + '.pickle')
        test_data = tud.DataLoader(DataSet(test_data, self.n_know), batch_size=batch_size, shuffle=True, num_workers=0)
        return test_data

    def load_n_cross_data(self, k, batch_size):
        print('loading cross_' + str(k) + ' train_val data ...')
        train_data = pd.read_pickle('data/train_val/cross_' + str(k) + '_train_' + self.dataset + '.pickle')
        val_data = pd.read_pickle('data/train_val/cross_' + str(k) + '_val_' + self.dataset + '.pickle')
        train_data = tud.DataLoader(DataSet(train_data, self.n_know), batch_size=batch_size, shuffle=True, num_workers=0)
        val_data = tud.DataLoader(DataSet(val_data, self.n_know), batch_size=batch_size, shuffle=True, num_workers=0)
        return train_data, val_data

    def get_data_shape(self):
        return self.n_know, self.n_stu, self.n_exer

    def get_data_group(self):
        return self.data_group
