import torch
import torch.nn as nn
import numpy as np
import os
import pandas as pd
# from tensorboardX import SummaryWriter

import argparse
import utils

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

# Device configuration
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)


def main(args):
    '''load dataset'''

    total_data = torch.FloatTensor(np.load(os.path.join('..', 'data', 'train_data.npy')))
    test_data = torch.FloatTensor(np.load(os.path.join('..', 'data', 'test_data.npy')))
    test_data = test_data[:, :args.using_data_num]

    data_len = int(total_data.shape[0])
    split_index = int(data_len * 0.8)

    train_data = total_data[:split_index]
    val_data = total_data[split_index:]

    x_train = train_data[:,: args.using_data_num].long()
    y_train = train_data[:,-1]

    x_val = val_data[:,: args.using_data_num].long()
    y_val = val_data[:,-1]

    final_train_data = TensorDataset(x_train, y_train)
    final_val_data = TensorDataset(x_val, y_val)
    final_val_data = TensorDataset(test_data)

    '''use embed'''
    if args.use_10k_embed:
        weights = utils.load_embedding()
    else:
        weights = None

    '''load_model'''
    if args.model == 'cnn':
        model = utils.simple_CNN(using_data_num=args.using_data_num,use_10k_embed= weights, device=device).to(device)
    elif args.model == 'lstm':
        model = utils.simple_LSTM(using_data_num=args.using_data_num,use_10k_embed= weights, device=device).to(device)

    '''set loss and optimizer'''
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(args.epoch):

        train_loader = DataLoader(final_train_data, batch_size=args.batch_size, shuffle=True)

        for i, train_batch in enumerate(train_loader):
            train_x_batch, train_y_batch = train_batch
            train_y_batch = train_y_batch.to(device)

            predict_y = model(train_x_batch)
            loss = criterion(predict_y, train_y_batch.float().reshape(-1, 1))
            acc = utils.accuracy(predict_y, train_y_batch.float().reshape(-1, 1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'[Train] Epoch {epoch} | Loss {loss} | Acc {acc}')
        #
        # writer.add_scalar(f'train/loss', loss, epoch)
        # writer.add_scalar(f'train/acc', acc, epoch)

    '''validation'''
    if epoch % 1 == 0:
        model.eval()
        with torch.no_grad():
            val_loader = DataLoader(final_val_data, batch_size=args.batch_size, shuffle=False)

            loss_list = []
            acc_list = []
            for i, val_batch in enumerate(val_loader):
                val_x_batch, val_y_batch = val_batch
                val_y_batch = val_y_batch.to(device)

                predict_y = model(val_x_batch)
                loss = criterion(predict_y, val_y_batch.float().reshape(-1, 1))
                acc = utils.accuracy(predict_y, val_y_batch.float().reshape(-1, 1))

                loss_list.append(loss), acc_list.append(acc)


    '''test'''
    final_predict = []
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)
    for i, test_batch in enumerate(test_loader):
        test_x_batch, test_y_batch = test_batch
        test_predict_y = model(val_x_batch)
        test_predict_y = torch.round(torch.sigmoid(test_predict_y))
        test_predict_y = np.array(test_predict_y.detach().cpu().reshape(-1), dtype='int')

        final_predict.append(test_predict_y)

    final_predict = np.concatenate(final_predict)

    pred_df = pd.DataFrame({
        'id': [i for i in range(len(final_predict))],
        'sentiment': final_predict
    })

    with open(os.path.join('predict.csv'), 'w') as f:
        f.write(pred_df.to_csv(index=False))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['cnn', 'lstm'], type=str)
    parser.add_argument('--use_10k_embed', action='store_true')
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--using_data_num', type=int, default=45)
    args = parser.parse_args()
    # writer = SummaryWriter()
    main(args)