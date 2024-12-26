from sklearn.model_selection import KFold
import pandas as pd
from argparse import ArgumentParser

def save_KFold_data(data, K, dataset_file):
    kf = KFold(n_splits=K)
    cross = 1
    for train_index, val_index in kf.split(data):
        train_data = data.iloc[train_index]
        val_data = data.iloc[val_index]
        train_data.to_pickle('data/train_val/cross_' + str(cross) +'_train_' + dataset_file + '.pickle')
        val_data.to_pickle('data/train_val/cross_' + str(cross) +'_val_' + dataset_file + '.pickle')
        print(dataset_file + ' cross_' + str(cross) +' train_val data saved...')
        cross += 1

def save_test_data(test_data, dataset_file):
    test_data.to_pickle('./data/test/test_' + dataset_file + '.pickle')
    print('test data saved...')

def load_data(dataset_file):
    data = pd.read_pickle('./data/' + dataset_file + '.pickle')
    max_rt = 60
    if dataset_file == "junyi":
        max_rt = 60
    elif dataset_file == "assistment2017":
        max_rt = 150
    elif dataset_file == "ednet":
        max_rt = 60
    elif dataset_file == "pisa2015":
        max_rt = 60
    data.loc[data['rt'] > max_rt, 'rt'] = max_rt
    data['rt'] = data['rt'].astype(int)
    return data

def init_data(data, train_rate):
    train_data = data.sample(int(len(data) * train_rate))
    test_data = data.drop(labels=train_data.index)
    return train_data, test_data

if __name__ == '__main__':
    parser = ArgumentParser("HONCD")
    # runtime args
    parser.add_argument("--dataset", type=str, help='pisa2015 / junyi / ednet / assistment2017', default="ednet")
    args = parser.parse_args()
    dataset_file = args.dataset
    data = load_data(dataset_file)
    train_data, test_data = init_data(data, 0.2)
    save_test_data(test_data, dataset_file)
    save_KFold_data(train_data, 5, dataset_file)



