import numpy as np
import pandas as pd
import torch


class PriceDataset:

    def __init__(self):
        self.train_x = None
        self.train_y = None
        self.val_x = None
        self.val_y = None

        train = pd.read_csv('data/price_data_tr.csv')
        val = pd.read_csv('data/price_data_val.csv')

        train['date'] = train['date'].apply(lambda x: str(x[:6])).astype('float')
        val['date'] = val['date'].apply(lambda x: str(x[:6])).astype('float')

        train_x, train_y = train.drop(['price'], axis=1), train[['price']]
        val_x, val_y = val.drop(['price'], axis=1), val[['price']]



        self.train_x, self.train_y = train_x, train_y
        self.val_x, self.val_y = val_x, val_y


    def getDataset(self):
        return [self.train_x, self.train_y, self.val_x, self.val_y]


if __name__ == '__main__':
    price_dataset = PriceDataset()
    [tr_x, tr_y, val_x, val_y] = price_dataset.getDataset()
    print(1)