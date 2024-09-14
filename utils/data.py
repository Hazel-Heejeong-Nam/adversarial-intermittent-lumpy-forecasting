import os
import numpy as np
from torchvision import transforms
from PIL import Image
import torch
import glob
import pickle
from utils.evaluate import *
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error


def load_train_test(args, data):

    np_data = data.values
    id = np_data[:,0]
    data = np_data[:, 1:]
    
    # train = data[:, :-(args.p + args.horizon)]
    train = data[:, :-args.horizon]
    
    if args.dataset =='m5':
        stride = 7
        num_window = (train.shape[-1] - (args.p + args.horizon))//stride + 1    
        print(f'{num_window} train windows are available')
        trainset = [np.concatenate([id.reshape(-1,1), train[:, (i*stride) :(i*stride) +args.p + args.horizon]], axis=-1) for i in range(num_window)]

    else :
        num_window = train.shape[-1] - (args.p + args.horizon)  + 1
        print(f'{num_window} train windows are available')
        trainset = [np.concatenate([id.reshape(-1,1), train[:, i:i+args.p + args.horizon]], axis=-1) for i in range(num_window)]
    
    trainset = np.concatenate(trainset, axis=0)    
    testset = data[:, -(args.p + args.horizon):]
    testset = np.concatenate([id.reshape(-1,1), testset], axis=-1)

    return trainset, testset

def load_train_test_hybrid(fname):
    if os.path.isfile(fname):
        with open(fname, 'rb') as file:
            res_dict = pickle.load(file)
        result = res_dict['res']
        test_X = res_dict['test_X']
        test_Y = res_dict['test_Y']
        test_stat = res_dict['test_pred']
    else :
        ValueError('Corresponding file does not exist')
        
    train_ids = result[0]['input'][:, 0]
    history = np.stack([item['input'][:, 1:] for item in result], axis=0).squeeze()
    X =np.stack( [item['output'][:, 1:] for item in result], axis=0).squeeze()
    label =np.stack( [item['gt'][:, 1:] for item in result], axis=0).squeeze()
    if len(history.shape)>1 :
        history = np.concatenate(history, axis=0)
        X = np.concatenate(X, axis=0)
        label = np.concatenate(label, axis=0)    
    train_data = {'ids': train_ids, 'history': history, 'X': X, 'label': label}
    
    test_ids = test_X[:, 0]
    test_data = {'ids': test_ids, 'history': test_X[:, 1:], 'X': test_stat[:, 1:], 'label': test_Y[:, 1:]}
    
    return  train_data, test_data

class ts_dataset(torch.utils.data.Dataset):
    def __init__(self, args ,data):
        self.ids = data[:, 0]
        data = data[:,1:]
        self.Y = torch.clamp(torch.from_numpy(np.array(data[:, -args.horizon : ], dtype=np.float32)), min=0)
        self.X = torch.clamp(torch.from_numpy(np.array(data[:, :args.p], dtype=np.float32)), min=0)
        assert self.Y.shape[-1] + self.X.shape[-1] == data.shape[-1]

    def __getitem__(self, idx):
        id = self.ids[idx]
        X = self.X[idx]
        Y = self.Y[idx]

        
        return id, X, Y

    def __len__(self):
        return len(self.ids)
    
class ts_dataset_hybrid(torch.utils.data.Dataset):
    def __init__(self, data):

        self.label = torch.clamp(torch.from_numpy(np.array(data['label'], dtype=np.float32)), min=0)
        self.history = torch.clamp(torch.from_numpy(np.array(data['history'], dtype=np.float32)), min=0)
        self.ids = data['ids']
        self.X = torch.clamp(torch.from_numpy(np.array(data['X'], dtype=np.float32)), min=0)


    def __getitem__(self, idx):
        #print(idx)
        id = self.ids[idx]
        history = self.history[idx]
        first_pred = self.X[idx]
        label = self.label[idx]
        
        data = torch.cat((history,first_pred), dim=0)   

        
        return id, data, label

    def __len__(self):
        return len(self.ids)
    
def read_as_nixtla(df):
    df_melted = df.melt(id_vars=['product_id'], var_name='month_year', value_name='y')

    df_melted['ds'] = pd.to_datetime(df_melted['month_year'], format='ISO8601')

    df_melted.drop(columns=['month_year'], inplace=True)
    df_melted = df_melted[['product_id', 'ds', 'y']]
    df_melted.columns = ['unique_id', 'ds', 'y']

    return df_melted

def read_data(model, data):
    df = pd.read_csv(f'data/{data}_as_mlusage.csv')
    freq = 'M' if (data=='auto' or data=='raf') else 'D'
    # if (data=='uci' or data=='raf'):
    #     stride = 2
    # elif data=='m5':
    #     stride = 5
    # elif data == 'auto':
    #     stride = 1

    if model == 'CrostonClassic': # or something else from nixtla    
        df = read_as_nixtla(df)
    if model == 'CrostonOptimized': # or something else from nixtla    
        df = read_as_nixtla(df)
    if model =='ADIDA':
        df = read_as_nixtla(df)
    if model == 'NBEATS':
        df = read_as_nixtla(df)
        
    return df, freq #, stride


def read_bdsi_nixtla(fname):
    df = pd.read_csv(fname)
    df_melted = df.melt(id_vars=['product_id'], var_name='month_year', value_name='y')

    df_melted['ds'] = pd.to_datetime(df_melted['month_year'], format='ISO8601')

    df_melted.drop(columns=['month_year'], inplace=True)
    df_melted = df_melted[['product_id', 'ds', 'y']]
    df_melted.columns = ['unique_id', 'ds', 'y']

    return df_melted

