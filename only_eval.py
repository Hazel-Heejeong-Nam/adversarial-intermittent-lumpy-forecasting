import pandas as pd
from utils.evaluate import * 
import os
import glob

def agg_result(f, ground_truth, pred, name):
        
    # round-up
    ground_truth = np.clip(np.trunc(ground_truth), a_min=0, a_max=None)
    pred = np.clip(np.trunc(pred), a_min=0, a_max=None)
    ground_truth = np.array(ground_truth, dtype=np.int32)
    pred = np.array(pred, dtype=np.int32)

    rmse = mean_squared_error(y_true=ground_truth, y_pred=pred, squared=False)
    # mae = mean_absolute_error(y_true=ground_truth, y_pred=pred)
    mape, _ = average_MAPE(ground_truth, pred)
    smape, _ = average_sMAPE(ground_truth, pred)
    # wape, _ = average_WAPE(ground_truth, pred)
    wmape, _ = average_wMAPE(ground_truth, pred)
    # spec75, _ = average_SPEC(ground_truth, pred, a1=0.75, a2=0.25)
    spec50, _ = average_SPEC(ground_truth, pred, a1=0.5, a2=0.5)
    # spec25, _ = average_SPEC(ground_truth, pred, a1=0.25, a2=0.75)
    std_mae = calculate_stddev(ground_truth, pred)
    trend_rmse = calculate_trend(ground_truth, pred)
    
    f.write("\n")
    f.write("---------- %s ---------\n" % (name))
    f.write("Test RMSE: %f \n" % (rmse))
    # f.write("Test MAE: %f \n" % (mae))
    f.write("mape: %f \n" % (mape))
    f.write("smape: %f \n" % (smape))
    # f.write("wape: %f \n" % (wape))
    f.write("wmape: %f \n" % (wmape))
    f.write("std mae: %f \n" % (std_mae))
    f.write("trend rmse: %f \n" % (trend_rmse))
    # f.write("spec (0.75, 0.25): %f \n" % (spec75))
    f.write("spec (0.5, 0.5): %f \n" % (spec50))
    # f.write("spec (0.25, 0.75): %f \n" % (spec25))
    f.write('----------------------------\n')
    f.write('\n')    
    
    
if __name__ == "__main__":
    path = 'file_jh3'
    file_list = os.listdir(path)

    txtfile = open('final_performances_jh3.txt', "w")
    
    for name in file_list:
        print(name)
        df_values = pd.read_csv(os.path.join(path, name)).values[1:, 1:]
        true = df_values[:, :df_values.shape[-1]//2]
        pred = df_values[:, df_values.shape[-1]//2:]
        agg_result(txtfile, true, pred, name)