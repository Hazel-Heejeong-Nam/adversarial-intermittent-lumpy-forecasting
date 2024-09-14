import pickle
import numpy as np
import matplotlib
import pandas as pd
import pickle
import os
import yaml
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
import random
matplotlib.use('agg')

def customer_chunk(data, customer):
    mask = [data[i,0].split('_')[0]==customer for i in range(data.shape[0])]
    new_data = data[mask,:]
    return new_data[:,1:], new_data[:,0]
    
def cross_normalized_distance(a,b, slide=3,direction='single'): # 작을수록 좋고 0~, a 가 target
    n = len(a)
    if direction =='single':
        list= [np.linalg.norm(a[s:] - b[:n-s], ord=None) for s in range(1,slide+1)]
    elif direction == 'both':
        list= [np.linalg.norm(a[s:] - b[:n-s], ord=None) for s in range(0,slide+1)] + [np.linalg.norm(b[s:] - a[:n-s]) for s in range(1, slide+1)]
    return min(list)/2 # scale to 0~1

def normalize_data(data, meta, seq_len):
    X_train_idxs, X_test_idxs = data.X_train[:, 0].unsqueeze(1), data.X_test[:, 0].unsqueeze(1) # exclude idxs before normalization
    if meta:
        X_train = data.X_train[:, 1:] 
        X_test = data.X_test[:, 1:]
    else:
        X_train = data.X_train[:, 1:seq_len+1] 
        X_test = data.X_test[:, 1:seq_len+1]
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train_scaled = torch.tensor(scaler.transform(X_train).astype(np.float32))
    X_test_scaled = torch.tensor(scaler.transform(X_test).astype(np.float32))
    X_train = torch.cat((X_train_idxs, X_train_scaled), dim=1)
    X_test = torch.cat((X_test_idxs, X_test_scaled), dim=1)
    data.X_train = X_train
    data.X_test = X_test

def find_labels(feature, data):
    if feature not in data.columns:
        return None
    labels = list(data[feature].unique())
    return sorted(labels)

def inf_generator(iterable):
    """Allows training with DataLoaders in a single infinite loop:
    for i, (x, y) in enumerate(inf_generator(train_loader)):
    """
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()

def get_r2_score(output, target):
    target_mean = torch.mean(target)
    sst = torch.sum((target - target_mean) ** 2)
    sse = torch.sum((target - output) ** 2)
    r2 = 1 - sse / sst
    return r2

def get_sse(model, test_x, test_y):
    preds = model(test_x)
    sum_squared_error = torch.sum((preds - test_y) ** 2)
    return sum_squared_error

class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val

def read_model(path):
    with open(path, 'rb') as f:
        data =pickle.load(f)
    try :
        a = data['model']
        return data['model'], data['config']
    except :
        return data, None

def load_config(path='./config.yaml'):
    with open(path, 'r') as file:
        return yaml.safe_load(file)

def encode_labels(data, cols):
    label_encoder = LabelEncoder()
    for col in cols:
        data[col] = label_encoder.fit_transform(data[col])
    return data

def split_data_by_proprietary(data):
    data_0 = data[data['ProductProprietary'] == 0]
    data_1 = data[data['ProductProprietary'] == 1]
    return data_0, data_1

def split_data_by_density(data, month_len, perc=50):
    threshold = int(month_len * (perc/100))
    data_0 = data[data['NonZeroCount'] <= threshold]
    data_1 = data[data['NonZeroCount'] > threshold]
    return data_0, data_1