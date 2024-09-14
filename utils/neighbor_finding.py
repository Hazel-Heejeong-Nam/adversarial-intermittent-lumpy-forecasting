from utils import cross_normalized_distance, customer_chunk
import numpy as np
import numpy as np
import matplotlib
import pandas as pd
from statsmodels.tsa.api import VAR
import warnings
from tqdm import trange, tqdm
import pickle
import os

def get_NCD_list(args, ncd_data): # 이건 data others
    ncd_dict = {}
    ids= ncd_data['Account'].astype(str) +'_'+ ncd_data['Prime'].astype(str) +'_'+ ncd_data['Discrete'].astype(str)
    np_u = ncd_data.iloc[:,:args.ts_len].values
    np_u = np.concatenate([ids.values.reshape(-1,1), np_u], axis=1)

    for target in tqdm(np_u):
        t = target[1:]
        target_id = target[0]
        scaled_target = t/np.sum(t) if np.sum(t)!=0 else t
        cust = target[0].split('_')[0]
        data_cust, ids_cust = customer_chunk(np_u, cust)
        k = args.k if len(data_cust)>args.k else len(data_cust)
        if len(data_cust)==1:
            continue

        vallist=[]
        # get normalized cross distance from all data in the customer group
        for idx, row in enumerate(data_cust):
            scaled_row = row/np.sum(row) if np.sum(row)!=0 else row
            val = cross_normalized_distance(np.squeeze(scaled_target)[:-1*args.out_len], np.squeeze(scaled_row)[:-1*args.out_len], slide =args.max_shift, direction=args.direction)
            vallist.append(round(val,6))
        sorted_indices = np.argsort(np.array(vallist))
        df = {'target' : t} # include itself
        for i in range(k):
            idx = sorted_indices[i]
            df['cand_'+str(i)] = data_cust[idx]

        ncd_dict[target_id] = df

    if not os.path.isdir('models/var/ckpt'):
        os.makedirs('models/var/ckpt', exist_ok=True)
    with open(f'models/var/ckpt/NCD_info_{args.idx_group}_{args.k}_{args.out_len}.pkl', 'wb') as f:
        pickle.dump(ncd_dict, f)

    return ncd_dict



def NCD_VAR(args, ncd_data, dict): # 이건 yes purchase
    print("VAR : Start Iteration") 
    warnings.filterwarnings("ignore")
    pred_else = []
    gt_else = []
    id_else = []

    errlist = []

    for i in trange(len(ncd_data)):
        target = ncd_data.iloc[i,:]
        t = target.iloc[:args.ts_len].values
        target_id = str(target['Account']) + '_' + str(target['Prime']) +  '_' +  str(target['Discrete'])

        
        try :
            df = dict[target_id]
        except :
            errlist.append(i)
            continue
        # Modify : original => diff
        mydata = pd.DataFrame(df)
        mydata_diff = mydata.diff().dropna()
        train = mydata.iloc[:-1*args.out_len,:]
        train_diff = mydata_diff.iloc[:-1*args.out_len,:]
        test = mydata.iloc[-1*args.out_len:,:]
        test_diff = mydata_diff.iloc[-1*args.out_len:,:]

        # for post processing
        max_sum_window = max([train['target'][i:i+args.var_outlen].sum() for i in range(len(train['target'])-args.var_outlen+1)])

        # fit model
        forecasting_model = VAR(train_diff.astype(float))
        minval = np.inf
        for p in range(1,args.p): # find the best look back period
            try : 
                results = forecasting_model.fit(p)
                if results.aic < minval:
                    minval, minp = results.aic, p
            except :
                pass 
        try :
            a = str(minp)
        except :
            errlist.append(i)
            continue
        results = forecasting_model.fit(minp)
        lagged_values = train_diff.values[-1*minp:]
        forecast = pd.DataFrame(results.forecast(y= lagged_values, steps=args.var_outlen), index = list(range(test_diff.index[0], test_diff.index[0]+args.var_outlen)), columns= mydata.columns) #차분

        # Restore : diff => original
        for col in forecast.columns:
            forecast[col] = mydata[col].iloc[-1*args.out_len-1] + forecast[col].cumsum()

        # post processing
        if args.postscale:
            pred = forecast['target'].values - np.min(forecast['target'].values) if np.min(forecast['target'].values)<0  else forecast['target'].values # or clippling
            if pred.sum() >max_sum_window: 
                pred = pred * max_sum_window / pred.sum() 
        else :
            pred = forecast['target'].values
        if len(pred)==0: # if error occurs
            errlist.append(i)
            continue
        else :
            pred_else.append(pred[args.lag:args.out_len])
            gt_else.append(target.iloc[args.ts_len-args.out_len+args.lag:args.ts_len].values)
            id_else.append(target_id)
    print(f'VAR : {len(errlist)} excluded')
    
    return id_else, np.array(pred_else), np.array(gt_else), ncd_data.iloc[errlist]
