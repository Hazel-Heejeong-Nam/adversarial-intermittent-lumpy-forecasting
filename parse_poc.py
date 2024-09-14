import argparse

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument("--gpu", default=3, type=int)
    parser.add_argument("--seed", default=50, type=int)
    
    parser.add_argument("--dataset", default='m5', choices=['m5', 'uci', 'raf', 'auto'])
    parser.add_argument("--save_csv", action='store_true')
    parser.add_argument("--horizon", default=28, type=int)
    parser.add_argument("--p", default=28, type=int)
    parser.add_argument("--base_model", default='poc')

    
    parser.add_argument("--f_type", default='recursive', choices=['recursive', 'linear'])
    parser.add_argument("--d_type", default='recursive', choices=['recursive', 'linear', 'None'])
    
    
    parser.add_argument("--lr_f", default=0.001, type=float)
    parser.add_argument("--lr_d", default=0.001, type=float)
    parser.add_argument("--alpha", default=0.5, type=float, help="weight of reconstruction loss")
    parser.add_argument("--epoch", default=100, type=int)
    parser.add_argument("--batch_size", default=256, type=int)
    args = parser.parse_args()

    return args