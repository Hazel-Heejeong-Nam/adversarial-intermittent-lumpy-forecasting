
from statsforecast import StatsForecast
from statsforecast.models import CrostonClassic, CrostonOptimized, ADIDA
from statsforecast.models import AutoARIMA, AutoETS, AutoCES, AutoRegressive, ETS
from neuralforecast import NeuralForecast
from neuralforecast.models import NBEATS
from tqdm import tqdm
import pickle
import pandas as pd
import numpy as np
import os
####
import warnings
warnings.filterwarnings("ignore")
####
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


def crostonclassic(df, freq, args):
    statsmodels = [CrostonClassic()]
    dates = df['ds'].unique()[-args.horizon:] # last 3 months
    dates_testX = df['ds'].unique()[-args.horizon-args.p:-args.horizon]
    # dates_testX = df['ds'].unique()[:-args.horizon]
    test_Y = convert_data(df.query('ds in @dates'))
    test_X = df.query('ds in @dates_testX')
    
    sf_test = StatsForecast(df = test_X, models = statsmodels, freq = freq,  n_jobs = -1)
    test_pred = sf_test.forecast(h=args.horizon)
    test_pred = test_pred.reset_index()
    test_pred = convert_to_model_array(test_pred,[args.base_model])
    id = test_Y[:,0]
        
    return id, np.round(np.array(test_Y[:,1:], dtype=np.float64)), np.round(np.array(test_pred[:,1:], dtype=np.float64))

def crostonoptimized(df, freq, args):
    statsmodels = [CrostonOptimized()]
    dates = df['ds'].unique()[-args.horizon:] # last 3 months
    dates_testX = df['ds'].unique()[-args.horizon-args.p:-args.horizon]
    test_Y = convert_data(df.query('ds in @dates'))
    test_X = df.query('ds in @dates_testX')
    
    sf_test = StatsForecast(df = test_X, models = statsmodels, freq = freq,  n_jobs = -1)
    test_pred = sf_test.forecast(h=args.horizon)
    test_pred = test_pred.reset_index()
    test_pred = convert_to_model_array(test_pred,[args.base_model])
    id = test_Y[:,0]
        
    return id, np.round(np.array(test_Y[:,1:], dtype=np.float64)), np.round(np.array(test_pred[:,1:], dtype=np.float64))


def adida(df, freq, args):
    statsmodels = [ADIDA()]
    dates = df['ds'].unique()[-args.horizon:] # last 3 months
    dates_testX = df['ds'].unique()[-args.horizon-args.p:-args.horizon]
    test_Y = convert_data(df.query('ds in @dates'))
    test_X = df.query('ds in @dates_testX')
    
    sf_test = StatsForecast(df = test_X, models = statsmodels, freq = freq,  n_jobs = -1)
    test_pred = sf_test.forecast(h=args.horizon)
    test_pred = test_pred.reset_index()
    test_pred = convert_to_model_array(test_pred,[args.base_model])
    id = test_Y[:,0]
        
    return id, np.round(np.array(test_Y[:,1:], dtype=np.float64)), np.round(np.array(test_pred[:,1:], dtype=np.float64))


def nbeats(df, freq, args):
    models = [NBEATS(input_size=args.p, h=args.horizon, max_steps=args.max_step, step_size=7)]
    dates = df['ds'].unique()[-args.horizon:] # last 3 months
    dates_testX = df['ds'].unique()[-args.horizon-args.p:-args.horizon]
    test_Y = convert_data(df.query('ds in @dates'))
    test_X = df.query('ds in @dates_testX')
    train_XY = df.query('ds not in @dates')
    
    nf = NeuralForecast(models = models,freq = freq)
    nf.fit(df=train_XY)
    test_pred = nf.predict()
    test_pred = test_pred.reset_index()
    test_pred = convert_to_model_array(test_pred,[args.base_model])
    id = test_Y[:,0]
    
    return id, np.round(np.array(test_Y[:,1:], dtype=np.float64)), np.round(np.array(test_pred[:,1:], dtype=np.float64))
