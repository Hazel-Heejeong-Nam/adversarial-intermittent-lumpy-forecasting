from utils.data import *

def lstm(args, data_df):
    train_data, test_data = load_train_test(args, data_df)
    train_dataset = ts_dataset(args, train_data)
    test_dataset = ts_dataset(args, test_data)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=4, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, num_workers=4, shuffle=False)
    
    
    
    
    
    return None