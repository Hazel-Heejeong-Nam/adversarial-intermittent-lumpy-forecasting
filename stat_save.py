from statsforecast import StatsForecast
from statsforecast.models import ADIDA, CrostonClassic, IMAPA, TSB
from neuralforecast.models import NBEATS
from statsforecast.models import AutoARIMA, AutoETS, AutoCES, AutoRegressive, ETS
import numpy as np
import neuralforecast
from utils.data import *
from tqdm import tqdm
from parse import parse_args
import pickle
import pandas as pd
import os
####
import warnings
warnings.filterwarnings("ignore")

def calculate_rolling_avg(x, window_size, months_out):
    x = np.array(x, dtype=np.int32)
    x = pd.DataFrame(x).rolling(window=window_size, min_periods=1, axis=1).mean().values
    return x[:, -months_out:]

def convert_to_model_array(df, model):
    print('converting forecasted results...')
    grouped = df.groupby('unique_id')[model].apply(lambda x: x.squeeze().tolist()).reset_index()
    result_list = np.column_stack((grouped['unique_id'], grouped[0].tolist()))

    return result_list

def convert_data(df):
    print('converting data')
    grouped = df.groupby('unique_id')['y'].apply(list).reset_index()
    result_arr = np.column_stack((grouped['unique_id'], grouped['y'].tolist()))

    return result_arr

def stat_forecast(args):
    if args.base_model=='MovingAverage':
        df = pd.read_csv(f'data/{args.dataset}_as_mlusage.csv')
        tslen = df.values.shape[-1] -1
    else:
        if args.base_model == 'ADIDA':
            statsmodels = [ADIDA()]
        elif args.base_model == 'CrostonClassic':
            statsmodels = [CrostonClassic()]
        df = read_bdsi_nixtla(f'data/{args.dataset}_as_mlusage.csv')
        freq = 'M' if (args.dataset=='auto' or args.dataset=='raf') else 'D'
        dates = df['ds'].unique()[-args.horizon:] # test Y
        dates_testX = df['ds'].unique()[-args.horizon-args.p:-args.horizon] # test X
        train_XY = df.query('ds not in @dates') # only exclude test Y
        tslen = len( df['ds'].unique())
        test_Y = convert_data(df.query('ds in @dates'))
        test_X = df.query('ds in @dates_testX')
    
    if args.dataset =='m5':
        stride = 7
        available_set_num = (tslen  - args.horizon - (args.p + args.horizon))//stride + 1    
        print('Avaliable number of window :', available_set_num)
    else:
        stride=1
        available_set_num = tslen - args.horizon - (args.p + args.horizon) + 1
        print('Avaliable number of window :', available_set_num)
        
        
    if args.base_model=='MovingAverage':
        content = df.values[:,1:]
        id = df.values[:,0].reshape(-1,1)
        test_Y = content[:,-args.horizon:]
        test_X = content[:, -(args.horizon + args.p): -args.horizon]
        test_pred = calculate_rolling_avg(test_X, window_size=args.p, months_out=args.horizon)
        train_XY = content[:, : -args.horizon]
        result = []
        for i in range(available_set_num):
            in_data = train_XY[:, i*stride : i*stride + args.p]
            gt_data = train_XY[:, i*stride + args.p:  i*stride + args.p+args.horizon]
            output_data = calculate_rolling_avg(in_data, window_size=args.p, months_out=args.horizon)
            converted_input = np.concatenate([id, in_data], axis=-1)
            converted_output = np.concatenate([id, output_data], axis=-1)
            converted_gt = np.concatenate([id, gt_data], axis=-1)
            result.append({'input': converted_input, 'output': converted_output, 'gt': converted_gt})
            print(converted_gt.shape)
        test_X = np.concatenate([id, test_X], axis=-1)
        test_Y = np.concatenate([id, test_Y], axis=-1)
        test_pred = np.concatenate([id, test_pred], axis=-1)

    else:
        sf_test = StatsForecast(df = test_X, models = statsmodels, freq = freq,  n_jobs = -1)
        test_pred = sf_test.forecast(h=args.horizon)
        test_pred = test_pred.reset_index()
        test_pred = convert_to_model_array(test_pred,[args.base_model])
        test_X = convert_data(test_X)
        starting_date = train_XY['ds'].min()
        X_offset = pd.DateOffset(months=args.p) if (args.dataset=='auto' or args.dataset=='raf') else pd.DateOffset(days=args.p) 
        Y_offset = pd.DateOffset(months=args.horizon) if (args.dataset=='auto' or args.dataset=='raf') else pd.DateOffset(days=args.horizon) 
        print('Start forecasting...')
        result = []
        for i in range(available_set_num):
            window = pd.DateOffset(months=i*stride) if (args.dataset=='auto' or args.dataset=='raf') else pd.DateOffset(days=i*stride) 
            train_X = train_XY[train_XY['ds']>=starting_date + window] 
            train_X = train_X[train_X['ds']<starting_date + window+ X_offset]
            train_Y = train_XY[train_XY['ds']>=starting_date + window+ X_offset]
            train_Y = train_Y[train_Y['ds']<starting_date + window+ X_offset + Y_offset]

            sf = StatsForecast(df = train_X, models = statsmodels, freq = freq,  n_jobs = -1)
            forecasts = sf.forecast(h=args.horizon)
            forecasts = forecasts.reset_index()
            converted_output = convert_to_model_array(forecasts,[args.base_model])
            converted_gt = convert_data(train_Y)
            converted_input = convert_data(train_X)
            result.append({'input': converted_input, 'output': converted_output, 'gt': converted_gt})

    if not os.path.isdir('ckpt'):
        os.makedirs('ckpt', exist_ok=True)
    fname = f'ckpt/{args.dataset}_{args.base_model}_p{args.p}_horizon{args.horizon}.pkl'
    with open(fname, 'wb') as file:
        pickle.dump({'res' : result, 'test_X': test_X, 'test_Y': test_Y, 'test_pred': test_pred}, file)



if __name__ =='__main__':
    args = parse_args()
    args.dataset='uci'
    args.p=18
    args.horizon=6
    args.base_model='MovingAverage'
    
    stat_forecast(args)
    
    