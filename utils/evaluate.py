import pandas as pd
import torch
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pickle
import os
import csv

# def mean_weighted_quantile_loss(y_pred: np.ndarray, y_true: np.ndarray, quantiles):
#     y_true_rep = y_true[:, None].repeat(len(quantiles), axis=1)
#     quantiles = np.array([float(q) for q in quantiles])
#     quantile_losses = 2 * np.sum(
#         np.abs(
#             (y_pred - y_true_rep)
#             * ((y_true_rep <= y_pred) - quantiles[:, None])
#         ),
#         axis=-1,
#     )  # shape [num_time_series, num_quantiles]
#     denom = np.sum(np.abs(y_true))  # shape [1]
#     weighted_losses = quantile_losses.sum(0) / denom  # shape [num_quantiles]
#     return weighted_losses.mean()


# def quantile_loss(y_true, y_pred, rho):
#     diff = y_true - y_pred
#     loss = np.where(diff > 0, rho * diff, (1 - rho) * -diff)
#     return loss

# def normalized_quantile_loss(y_true: np.ndarray, y_pred: np.ndarray, rho: float) -> float:
#     quantile_losses = quantile_loss(y_true, y_pred, rho)
#     numerator = 2 * np.sum(quantile_losses)
#     denominator = np.sum(np.abs(y_true))
    
#     # Calculate normalized quantile loss (rho-risk)
#     rho_risk = numerator / denominator if denominator != 0 else 0
#     return rho_risk

def average_SPEC(y_true, y_pred, a1=0.75, a2 = 0.25):
    assert y_true.shape == y_pred.shape
    spec_sum = 0
    for true, pred in zip(y_true, y_pred):
        for n in range(y_true.shape[1]):
            for m in range(n):
                a1_term = a1*np.minimum(true[m], true[:m].sum()-pred[:n].sum())
                a2_term = a2*np.minimum(pred[m], pred[:m].sum()-true[:n].sum())
                spec = (n-m+1)*np.maximum(0, np.maximum(a1_term, a2_term))
                spec_sum += spec
    avg_spec = spec_sum / (y_true.shape[0]*y_true.shape[1])
    return avg_spec , y_true.shape[0]

def calculate_stddev(y_true, y_pred):
    assert y_true.shape == y_pred.shape
    std_true = np.std(y_true, axis=-1)
    std_pred = np.std(y_pred, axis=-1)
    std_mae = mean_absolute_error(y_true = std_true, y_pred = std_pred)
    return std_mae

def calculate_trend(y_true, y_pred):
    t_true = np.sign(np.diff(y_true, axis=-1)).astype(int)
    t_pred = np.sign(np.diff(y_pred, axis=-1)).astype(int)
    rmse = mean_squared_error(y_true=t_true, y_pred=t_pred, squared=False)
    return rmse
    
def average_MAPE(y_true, y_pred):
    assert y_true.shape == y_pred.shape
    partsum=0
    cnt=0
    for true, pred in zip(y_true, y_pred):
        mask = true > 0
        if sum(mask) == 0 : # 한 row가 통째로 0이면 masked가 없음
            pass
        else : 
            masked_y_true = true[mask]
            masked_y_pred = pred[mask]
            error_ratio= np.mean(np.abs((masked_y_true - masked_y_pred) / masked_y_true))
            error_ratio = np.clip(error_ratio, 0, 1)
            partsum += error_ratio
            cnt +=1
    mape = partsum / cnt
    return mape, cnt

def average_sMAPE(y_true, y_pred):
    assert y_true.shape == y_pred.shape
    partsum=0
    cnt=0
    for true, pred in zip(y_true, y_pred):
        mask = (true+pred) > 0 
        if sum(mask) == 0 : # 한 row가 통째로 0이면 masked가 없음
            pass
        else : 
            true = true[mask]
            pred = pred[mask]
            error_ratio= np.mean(2 * np.abs(true - pred) / (np.abs(true) + np.abs(pred)))
            error_ratio = np.clip(error_ratio, 0, 2)
            partsum += error_ratio
            cnt +=1
    smape = partsum / cnt
    return smape, cnt

def average_WAPE(y_true, y_pred):
    assert y_true.shape == y_pred.shape
    row_sums = y_true.sum(axis=1)
    y_true = y_true[row_sums != 0]
    y_pred = y_pred[row_sums != 0]
    
    partsum=0
    for gt, pred in zip(y_true,y_pred):
        error = np.abs(pred - gt)
        error_ratio = np.divide(np.sum(error), np.sum(gt))
        error_ratio = np.clip(error_ratio, 0, 1)
        partsum += error_ratio
    wape = partsum / y_true.shape[0]
    return wape,  y_true.shape[0]

def average_wMAPE(y_true, y_pred):
    assert y_true.shape == y_pred.shape
    row_sums = y_true.sum(axis=1)
    y_true = y_true[row_sums != 0]
    y_pred = y_pred[row_sums != 0]
    
    partsum=0
    for gt, pred in zip(y_true,y_pred):
        weighted_error = np.multiply(gt, np.abs(pred - gt))
        error_ratio = np.divide(np.sum(weighted_error), np.sum(np.multiply(gt, gt)))
        error_ratio = np.clip(error_ratio, 0, 1)
        partsum += error_ratio
    wape = partsum / y_true.shape[0]
    return wape,  y_true.shape[0]

def RMSSE(y_true, y_pred, y_train):
    scale = np.mean(np.abs(np.diff(y_train, n=1)), axis=-1)
    y_true = y_true[scale != 0]
    y_pred = y_pred[scale != 0]
    scale = scale[scale != 0]
    rmsse = np.mean( np.mean((y_true - y_pred) ** 2, axis=-1) / scale )
    return rmsse


    

def write_result(fname, ground_truth, pred):

    if isinstance(fname, str):
        fname = os.path.join('performance',fname + '.txt')
        f = open(fname, "w")
    else :
        f = fname
        
    # round-up
    ground_truth = np.clip(np.trunc(ground_truth), a_min=0, a_max=None)
    pred = np.clip(np.trunc(pred), a_min=0, a_max=None)
    
    rmse = mean_squared_error(y_true=ground_truth, y_pred=pred, squared=False)
    mae = mean_absolute_error(y_true=ground_truth, y_pred=pred)
    mape, _ = average_MAPE(ground_truth, pred)
    smape, _ = average_sMAPE(ground_truth, pred)
    wape, _ = average_WAPE(ground_truth, pred)
    wmape, _ = average_wMAPE(ground_truth, pred)
    spec75, _ = average_SPEC(ground_truth, pred, a1=0.75, a2=0.25)
    spec50, _ = average_SPEC(ground_truth, pred, a1=0.5, a2=0.5)
    spec25, _ = average_SPEC(ground_truth, pred, a1=0.25, a2=0.75)
    std_mae = calculate_stddev(ground_truth, pred)
    
    f.write("\n")
    f.write("--------------------\n")
    f.write("Test RMSE: %f \n" % (rmse))
    f.write("Test MAE: %f \n" % (mae))
    f.write("mape: %f \n" % (mape))
    f.write("smape: %f \n" % (smape))
    f.write("wape: %f \n" % (wape))
    f.write("wmape: %f \n" % (wmape))
    f.write("spec (0.75, 0.25): %f \n" % (spec75))
    f.write("spec (0.5, 0.5): %f \n" % (spec50))
    f.write("spec (0.25, 0.75): %f \n" % (spec25))
    f.write('----------------------------\n')
    f.write('\n')    

def write_csv(fname ,id, true, pred):# create forecasting result
    true = np.clip(np.trunc(true), a_min=0, a_max=None)
    pred = np.clip(np.trunc(pred), a_min=0, a_max=None)
    fname = os.path.join('file',fname + '.csv')
    header = np.array(['product_id']+['true' for i in range(true.shape[1])]+['pred' for j in range(pred.shape[1])])
    data = np.concatenate([np.array(id).reshape(-1,1), true, pred], axis=-1)    
    df = pd.DataFrame(np.concatenate([header.reshape(1, -1), data], axis=0))
    df.to_csv(fname,  encoding='utf-8',header=False, index=None)
