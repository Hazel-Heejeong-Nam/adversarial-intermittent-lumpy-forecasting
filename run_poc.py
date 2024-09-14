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
from parse_poc import parse_args
import warnings
warnings.filterwarnings("ignore")
from tqdm import trange, tqdm
import random
from models import MLP_Discriminator, LSTM_Discriminator, MLP_forecaster, LSTM_forecaster

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
torch.autograd.set_detect_anomaly(True)

def train(args, train_loader, forecaster, discriminator, f, device):
    adversarial_loss = nn.BCELoss().to(device)
    supervised_loss = nn.MSELoss().to(device)
    feature_matching_loss = nn.MSELoss(reduction='sum').to(device)
        
    optimizer_F = torch.optim.Adam(forecaster.parameters(), lr=args.lr_f, betas=(0.9, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr_d, betas=(0.9, 0.999))

    n_epochs = args.epoch
    start_time = time.time()
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available else torch.FloatTensor

    for epoch in range(args.epoch):
        pred_list = []
        gt_list = []
        start = time.time()
        for i, (id, x, label) in tqdm(enumerate(train_loader)): # x : history
            if args.d_type == 'recursive':
                d_state = discriminator.init_hidden(x.shape[0], device)

            batch = torch.cat((x, label), dim=1).to(device) # scaling을 위해 잠시 결합
            scaler = torch.max(batch, dim=1).values.reshape(-1,1).to(device)
            scaled_batch = torch.nan_to_num(torch.div(batch, scaler), nan=0)
            scaled_X = scaled_batch[:, :x.shape[1]] 
            scaled_label = scaled_batch[:, x.shape[1]:] 

            real = Variable(Tensor(scaled_X.shape[0], 1).fill_(1.0), requires_grad=False).to(device)
            fake = Variable(Tensor(scaled_X.shape[0], 1).fill_(0.0), requires_grad=False).to(device)

            real_inputs = Variable(scaled_label.type(Tensor)) 
            optimizer_F.zero_grad()
            
            pred = forecaster(scaled_X)    
            forecast_loss = supervised_loss(pred, scaled_label)

            if args.d_type =='recursive':
                d_logits_pred, _ = discriminator(pred,d_state)
                f_loss = adversarial_loss(d_logits_pred.reshape(-1,1), real) + args.alpha * forecast_loss

            else :
                d_logits_pred = discriminator(pred)
                f_loss = adversarial_loss(d_logits_pred.reshape(-1,1), real) + args.alpha * forecast_loss
                

            
            f_loss.backward(retain_graph=True)
            optimizer_F.step()

            optimizer_D.zero_grad()
            if args.d_type =='recursive':
                real_loss = adversarial_loss(discriminator(real_inputs, d_state)[0].reshape(-1,1), real)
                fake_loss = adversarial_loss(discriminator(pred.detach(), d_state)[0].reshape(-1,1), fake)
            else : 
                real_loss = adversarial_loss(discriminator(real_inputs), real)
                fake_loss = adversarial_loss(discriminator(pred.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2

            d_loss.backward()
            optimizer_D.step()
            for p in discriminator.parameters():
                p.data.clamp_(-0.1, 0.1)
            pred_list.append((pred * scaler).detach().cpu().numpy())
            gt_list.append(label.detach().numpy())
            
            
        train_gt = np.concatenate(gt_list, axis=0)
        train_pred = np.concatenate(pred_list, axis=0)
        rmse = mean_squared_error(y_true=train_gt, y_pred=train_pred, squared=False)
        mae = mean_absolute_error(y_true=train_gt, y_pred=train_pred) 
        
        t =time.time() -start
        print(f'epoch {epoch}, time elapsed: {t}')
        f.write("[Epoch %d / %d] [rmse : %f][mae : %f][D loss: %.6f] [F loss: %.6f]\n" % (epoch, n_epochs, rmse, mae, d_loss.item(), f_loss.item()))
        
    return forecaster

def test(args, test_loader, forecaster, f, device):
    pred = []
    gt = []
    ids = []
    with torch.no_grad():
        for i, (id, x, label) in enumerate(test_loader): # x : history
            x = x.to(device)
            scaler = torch.max(x, dim=1).values.reshape(-1,1).to(device) # batch로 scaling하면 치팅임
            scaled_X = torch.nan_to_num(torch.div(x, scaler), nan=0).to(device)
            final_pred = forecaster(scaled_X)
            pred.append((final_pred * scaler).detach().cpu().numpy())
            gt.append(label.detach().numpy())        
            ids += list(id)
            
        test_gt = np.concatenate(gt, axis=0)
        test_pred = np.clip(np.concatenate(pred, axis=0), a_min=0, a_max=None)
        rmse = mean_squared_error(y_true=test_gt, y_pred=test_pred, squared=False)
        mae = mean_absolute_error(y_true=test_gt, y_pred=test_pred)    
        f.write("TEST | [rmse : %f][mae : %f]\n" % (rmse, mae))
        
    return test_gt, test_pred, ids


def main_worker(args):
    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'

    file_path = f'poc/{args.dataset}/F_{args.f_type}_D_{args.d_type}'
    os.makedirs('performance/'+file_path, exist_ok=True)
    os.makedirs('file/'+file_path, exist_ok=True)
    fname = os.path.join(file_path, f'lag{args.p}_fh{args.horizon}_flr_{args.lr_f}_dlr_{args.lr_d}_a{args.alpha}_epoch_{args.epoch}_bs_{args.batch_size}')
    f =  open(os.path.join('performance', fname+'.txt'), 'w')
    
    data_df, _ = read_data(args.base_model, args.dataset)
    train_data, test_data = load_train_test(args, data_df)
    train_dataset = ts_dataset(args, train_data)
    test_dataset = ts_dataset(args, test_data)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=4, shuffle=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, num_workers=4, shuffle=False, drop_last=True)
    
    
    if args.f_type =='recursive':
        forecaster = LSTM_forecaster(seq_length=args.horizon, batch_size=args.batch_size, device=device, num_classes=1, input_size=1, hidden_size=256, num_layers=2).to(device)
    elif args.f_type=='linear':
        forecaster = MLP_forecaster(in_shape=args.p, out_shape=args.horizon, num_layer=3, hidden_units=256).to(device)
        
    if args.d_type=='recursive':
        discriminator = LSTM_Discriminator(batch_size=args.batch_size, device=device, num_feats=1).to(device)
    elif args.d_type=='linear':
        discriminator = MLP_Discriminator(shape=args.horizon, hidden_units=16).to(device)
    
    forecaster = train(args, train_loader, forecaster, discriminator, f, device)
    true, pred, id = test(args, test_loader, forecaster,f, device)
    


    write_result(f, true, pred)
    if args.save_csv:
        write_csv(fname ,id, true, pred)
    
    
if __name__ =='__main__':
    args = parse_args()
    args.save_csv=True
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    main_worker(args)
    
    
