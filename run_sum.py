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
from tqdm import trange, tqdm
import random
from models import MLP_Discriminator, LSTM_Discriminator, MLP_forecaster, LSTM_forecaster

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
torch.autograd.set_detect_anomaly(True)



def train(args, forecaster, converter, discriminator, sum_estimator, train_loader, f, device):
    adversarial_loss = nn.BCELoss().to(device)
    supervised_loss = nn.MSELoss().to(device)
    # feature_matching_loss = nn.MSELoss(reduction='sum').to(device)
        
    optimizer_F = torch.optim.Adam(forecaster.parameters(), lr=args.lr_f, betas=(0.9, 0.999))
    optimizer_C = torch.optim.Adam(converter.parameters(), lr=args.lr_c, betas=(0.9, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr_d, betas=(0.9, 0.999))
    optimizer_S = torch.optim.Adam(sum_estimator.parameters(), lr=args.lr_s, betas=(0.9, 0.999))

    n_epochs = args.epoch
    start_time = time.time()
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available else torch.FloatTensor

    for epoch in range(args.epoch):
        pred_list = []
        gt_list = []
        sum_list = []
        start = time.time()
        for i, (id, x, label) in tqdm(enumerate(train_loader)): # x : history
            # if hybrid, x is concatenated form of history and the first prediction from given statistical model 
            
            batch = torch.cat((x, label), dim=1).to(device) 
            scaler = torch.max(batch, dim=1).values.reshape(-1,1).to(device) 
            scaled_batch = torch.nan_to_num(torch.div(batch, scaler), nan=0)
            scaled_X = scaled_batch[:, :x.shape[1]] 
            scaled_label = scaled_batch[:, x.shape[1]:] 
            scaled_label_sum = torch.sum(scaled_label, dim = -1)


            # for adversarial learning
            real = Variable(Tensor(scaled_X.shape[0], 1).fill_(1.0), requires_grad=False).to(device)
            fake = Variable(Tensor(scaled_X.shape[0], 1).fill_(0.0), requires_grad=False).to(device)
            real_inputs = Variable(scaled_label.type(Tensor)) 
            
            # PHASE1
            optimizer_F.zero_grad()
            optimizer_C.zero_grad()
            first_pred = forecaster(scaled_X) # shape[-1] -> latent dim
            pred = converter(first_pred)    # shape[-1] -> forecasting horizon
            forecast_loss = supervised_loss(pred, scaled_label)
            d_logits_pred = discriminator(pred)
            fc_loss = adversarial_loss(d_logits_pred.reshape(-1,1), real) + args.alpha * forecast_loss
            fc_loss.backward(retain_graph=True)
            # 이거 gradient를........... optimizer를 하나를 써야하나? back prop 을 두 번 해야하나?
            optimizer_C.step()
            optimizer_F.step()
            
            
            #PHASE2
            optimizer_F.zero_grad()
            optimizer_S.zero_grad()
            sum_pred = sum_estimator(first_pred)
            sum_loss = supervised_loss(sum_pred, scaled_label_sum)
            sum_loss.backward(retain_graph=True)
            optimizer_F.step()
            optimizer_S.step()
            
            
            # PHASE3
            optimizer_D.zero_grad()
            real_loss = adversarial_loss(discriminator(real_inputs), real)
            fake_loss = adversarial_loss(discriminator(pred.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()
            for p in discriminator.parameters():
                p.data.clamp_(-0.1, 0.1)
            pred_list.append((pred * scaler).detach().cpu().numpy())
            sum_list.append((sum_pred * scaler).detach().cpu().numpy())
            gt_list.append(label.detach().numpy())
            
            
            
            
        train_gt = np.concatenate(gt_list, axis=0)
        train_pred = np.concatenate(pred_list, axis=0)
        # train_sum = np.concatenate(sum_list, axis=0)
        ##########이걸로 뭐볼까
        
        rmse = mean_squared_error(y_true=train_gt, y_pred=train_pred, squared=False)
        mae = mean_absolute_error(y_true=train_gt, y_pred=train_pred) 
        
        t =time.time() -start
        print(f'epoch {epoch}, time elapsed: {t}')
        f.write("[Epoch %d / %d] [rmse : %f][mae : %f][D loss: %.6f] [FC loss: %.6f] [S loss : %.6f]\n" % (epoch, n_epochs, rmse, mae, d_loss.item(), fc_loss.item(), sum_loss.item()))
        
    return forecaster, converter

def test(args, test_loader, forecaster, converter,  f, device):
    pred = []
    gt = []
    ids = []
    with torch.no_grad():
        for i, (id, x, label) in enumerate(test_loader): # x : history
            x = x.to(device)
            scaler = torch.max(x, dim=1).values.reshape(-1,1).to(device) # batch로 scaling하면 치팅임
            scaled_X = torch.nan_to_num(torch.div(x, scaler), nan=0).to(device)
            first_pred = forecaster(scaled_X)
            final_pred = converter(first_pred)
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

    file_path = f'ours/{args.dataset}/sum_{args.model_subtype}'
    os.makedirs('performance/'+file_path, exist_ok=True)
    os.makedirs('file/'+file_path, exist_ok=True)
    fname = os.path.join(file_path, f'f_{args.lr_f}_c_{args.lr_c}_s_{args.lr_s}_d_{args.lr_d}_a{args.alpha}_epoch_{args.epoch}_lag{args.p}_fh{args.horizon}_latent_{args.latent_dim}_bs_{args.batch_size}')
    f =  open(os.path.join('performance', fname+'.txt'), 'w')
    discriminator = MLP_Discriminator(shape=args.horizon, hidden_units=16).to(device)
    
    if (args.model_subtype =='CrostonClassic') or (args.model_subtype =='MovingAverage'): # 이거 m5 조심
        fn = f'ckpt/{args.dataset}_{args.model_subtype}_p{args.p}_horizon{args.horizon}.pkl'
        
        # load data
        train_data, test_data = load_train_test_hybrid(fn)
        train_dataset = ts_dataset_hybrid(train_data)
        test_dataset = ts_dataset_hybrid(test_data)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=4, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, num_workers=4, shuffle=False)
    
        # init model
        forecaster = MLP_forecaster(in_shape=(args.horizon+args.p), out_shape=args.latent_dim, num_layer=2, hidden_units=32).to(device)
        
    elif args.model_subtype == 'recursive':
        
        # load data
        data_df, _ = read_data(args.base_model, args.dataset)
        train_data, test_data = load_train_test(args, data_df)
        train_dataset = ts_dataset(args, train_data)
        test_dataset = ts_dataset(args, test_data)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=4, shuffle=True, drop_last=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, num_workers=4, shuffle=False, drop_last=True)
        
        # init model
        forecaster = LSTM_forecaster(seq_length=args.latent_dim, batch_size=args.batch_size, device=device, num_classes=1, input_size=1, hidden_size=256, num_layers=2).to(device)     
    # 공통
    converter = MLP_forecaster(in_shape=args.latent_dim, out_shape=args.horizon, num_layer=2, hidden_units=128).to(device)
    sum_estimator = MLP_forecaster(in_shape=args.latent_dim, out_shape=1, num_layer=2, hidden_units=32).to(device)
    forecaster, converter  = train(args, forecaster, converter, discriminator, sum_estimator, train_loader, f, device)
    true, pred, id = test(args, test_loader, forecaster,converter, f, device)



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
    
    
