import torch
import torch.nn as nn
import pickle
import numpy as np
from torch.autograd import Variable
import time
from sklearn.metrics import mean_squared_error, mean_absolute_error
from models import *
from utils.data import *
from utils.evaluate import * 
from parse import parse_args
import warnings
warnings.filterwarnings("ignore")
from tqdm import trange
from models import *


os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
device = 'cuda' if torch.cuda.is_available() else 'cpu'



def main_worker(args):
    file_path = f'baselines/{args.dataset}/{args.base_model}'
    os.makedirs('performance/'+file_path, exist_ok=True)
    os.makedirs('file/'+file_path, exist_ok=True)
       
    
    data_df, freq = read_data(args.base_model, args.dataset)
    
    if args.base_model =="CrostonClassic" :
        fname = os.path.join(file_path, f'lag{args.p}_fh{args.horizon}')
        id, true, pred = crostonclassic(data_df, freq, args)
    elif args.base_model =="CrostonOptimized" :
        fname = os.path.join(file_path, f'lag{args.p}_fh{args.horizon}')
        id, true, pred = crostonoptimized(data_df, freq, args)
    elif args.base_model=="NBEATS":
        fname = os.path.join(file_path, f'lag{args.p}_fh{args.horizon}')
        id, true, pred = nbeats(data_df, freq, args)
    elif args.base_model=="ADIDA":
        fname = os.path.join(file_path, f'lag{args.p}_fh{args.horizon}')
        id, true, pred = adida(data_df, freq, args)
    elif args.base_model == 'rnn':
        pass
    elif args.base_model == 'lstm':
        pass

    
    write_result(fname, true, pred)
    if args.save_csv:
        write_csv(fname ,id, true, pred)

    
if __name__ =='__main__':
    args = parse_args()
    
    
    ## debugging
    args.save_csv=True
    args.base_model='NBEATS'
    args.dataset='m5'
    args.p=28
    args.horizon=28
    args.max_step=23900
    ########
    main_worker(args)
    
    
