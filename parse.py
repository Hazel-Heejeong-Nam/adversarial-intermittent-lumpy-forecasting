import argparse

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--gpu", default=3, type=int)
    parser.add_argument("--seed", default=0, type=int)
    
    
    parser.add_argument("--dataset", default='raf', choices=['m5', 'uci', 'raf', 'auto'])
    parser.add_argument("--save_csv", action='store_true')
    parser.add_argument("--alpha", default=0.5, type=float, help="weight of reconstruction loss")
    parser.add_argument("--epoch", default=100, type=int)
    
    parser.add_argument("--horizon", default=6, type=int)
    parser.add_argument("--p", default=18, type=int)
    parser.add_argument("--latent_dim", default=128)
    
    parser.add_argument("--model_subtype", default='MovingAverage', choices=['recursive', 'CrostonClassic', 'MovingAverage'])
    #run_sum.py
    parser.add_argument("--lr_f", default=0.001, type=float)
    parser.add_argument("--lr_c", default=0.001, type=float)
    parser.add_argument("--lr_s", default=0.001, type=float)
    parser.add_argument("--lr_d", default=0.001, type=float)

    
    # run_baselines.py
    parser.add_argument("--base_model", default='ours', choices=["CrostonClassic", "CrostonOptimized", "ADIDA","NBEATS",'ours', 'MovingAverage'])
    parser.add_argument("--max_step", type=int, default=1000)

    parser.add_argument("--batch_size", default=512, type=int)
    args = parser.parse_args()

    return args