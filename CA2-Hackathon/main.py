import argparse
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter

from model import *
from utils import binary_acc, load_pretrained_embedding

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

import pandas as pd
import numpy as np
import os


def main(args, writer):
    # Load dataset
    total_data = torch.tensor(np.load(os.path.join('.', 'data', 'train.npy')))
    X_test = torch.tensor(np.load(os.path.join('.', 'data', 'test.npy'))).long()
    X_test = X_test[:, :args.max_length]
    dataset_test = TensorDataset(X_test)

    shuffled = np.random.permutation(total_data.shape[0])
    train_index = shuffled[ :int(len(shuffled)*0.8)]
    val_index = shuffled[int(len(shuffled)*0.2): ]

    train_data = total_data[train_index]
    val_data = total_data[val_index]

    X_train = train_data[:, :args.max_length].long()  # [Batch, 50 Number of int (0~9999)]
    y_train = train_data[:, -1]  # [Batch, Y_data]
    dataset_train = TensorDataset(X_train, y_train)

    X_val = val_data[:, :args.max_length].long()  # [Batch, 50 Number of int (0~9999)]
    y_val = val_data[:, -1]  # [Batch, Y_data]
    dataset_val = TensorDataset(X_val, y_val)



    if args.pretrained_embed:
        weights = load_pretrained_embedding()
        print('Load pretrained weights:', weights.shape)
    else:
        weights = None
        print('Load initialized weights')

    if args.model == 'cnn':
        model = CNNBasedModel(max_length=args.max_length, pretrained_embedding=weights, device=args.device).to(args.device)
    elif args.model == 'lstm':
        model = LSTMBasedModel(lstm_length=args.max_length, pretrained_embedding=weights, device=args.device).to(args.device)
    elif args.model == 'bilstm':
        model = BiLSTMModel(lstm_length=args.max_length, device=args.device).to(args.device)

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.BCEWithLogitsLoss()

    # Repeat for 100 epochs
    for epoch in range(args.epoch):

        dataloader = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True)

        for batch_idx, samples in enumerate(dataloader):
            x_train, y_train = samples
            y_train = y_train.to(args.device)

            y_pred = model(x_train)

            # Update with huber-loss
            loss = criterion(y_pred, y_train.float().reshape(-1, 1))
            acc = binary_acc(y_pred, y_train.float().reshape(-1, 1))

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            # After update, we calculate the real value for prediction
            # Write log to tensorboard
        writer.add_scalar(f'train/loss', loss, epoch)
        writer.add_scalar(f'train/acc', acc, epoch)

        print(f'[Train] Epoch {epoch} | Loss {loss} | Acc {acc}')

        if epoch % 1 == 0:
            model.eval()
            with torch.no_grad():
                dataloader = DataLoader(dataset_val, batch_size=args.batch_size, shuffle=True)

                losses, accs = list(), list()
                for batch_idx, samples in enumerate(dataloader):
                    x_val, y_val = samples
                    y_val = y_val.to(args.device)

                    y_pred = model(x_val)

                    # Update with huber-loss
                    loss = criterion(y_pred, y_val.float().reshape(-1, 1)).item()
                    acc = binary_acc(y_pred, y_val.float().reshape(-1, 1)).item()

                    losses.append(loss), accs.append(acc)

            writer.add_scalar(f'val/loss', sum(losses) / len(losses), epoch)
            writer.add_scalar(f'val/acc', sum(accs) / len(accs), epoch)

            print(f'[Validation] Epoch {epoch} | Loss {sum(losses) / len(losses)} | Acc {sum(accs) / len(accs)}')

            model.train()

    # Prediction Test-set
    model.eval()

    predictions = list()

    dataloader = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False)
    for batch_idx, samples in enumerate(dataloader):
        x_test = samples[0]

        test_y_pred = model(x_test)
        test_y_pred = torch.round(torch.sigmoid(test_y_pred))
        test_y_pred = np.array(test_y_pred.detach().cpu().reshape(-1), dtype='int')

        predictions.append(test_y_pred)

    predictions = np.concatenate(predictions)

    pred_df = pd.DataFrame({
        'id': [i for i in range(len(predictions))],
        'sentiment': predictions
    })

    with open(os.path.join('predict.csv'), 'w') as f:
        f.write(pred_df.to_csv(index=False))

    print('Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    # Ablation study: w/ and w/o feature extraction type
    parser.add_argument('--model', choices=['cnn', 'lstm', 'bilstm'], type=str, help='Model for feature extraction')

    # Ablation study: w/ and w/o pretrained embedding
    parser.add_argument('--pretrained_embed', action='store_true', help='Use the pretrained embedding weight')
    # Pretrained: Kaggle provided
    # Ours: Word2Vec embedding we created

    # Ablation study: max-length of word in a sentence (95% rate on sentence lengths)
    parser.add_argument('--max_length', type=int, default=3773, help='Max word counts use as input')

    parser.add_argument('--batch_size', type=int, default=128, help='Batch-size')
    parser.add_argument('--epoch', type=int, default=10, help='Train epoch')
    parser.add_argument('--device', type=str, default='cuda', help='Cuda device (cuda:1)')

    args = parser.parse_args()

    writer = SummaryWriter('ca2_lstm_embed_re')
    main(args, writer)
