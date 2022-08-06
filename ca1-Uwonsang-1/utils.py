import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from scipy.stats import zscore
from uszipcode import SearchEngine
from sklearn.decomposition import PCA


def preprocessing(data_x, data_y, type='train'):

    total_data = pd.concat((data_x, data_y), axis=1)

    #remove outlier
    if type == 'train':
        total_data.drop(total_data.loc[total_data['sqft_living'] > 10000].index, axis=0, inplace=True)

    total_origin_data =total_data.copy()

    # remove column
    del total_data['id']
    del total_data['date']


    # # Feature engineering
    total_data['total_score'] = total_data['condition'] + total_data['grade'] + total_data['view']
    total_data['living_lot_ratio'] = total_data['sqft_living'] / total_data['sqft_lot']
    total_data['sqft_total_size'] = total_data['sqft_living'] + total_data['sqft_lot']
    total_data['total_room'] = total_data['bedrooms'] + total_data['bathrooms']
    total_data['above_living'] = total_data['sqft_above'] / total_data['sqft_living']

    contious_columns = total_data.columns.drop(['cluster', 'price', 'condition',
                                                'view', 'floors', 'zipcode', 'yr_renovated'
                                                ])

    ### contious_data
    log_columns = contious_columns.drop(['lat', 'long'])
    z_columns = ['lat', 'long']

    # log & zscore contious value
    for c in log_columns:
        total_data[c] = zscore(np.log1p(total_data[c].values))
    for c in z_columns:
        total_data[c] = zscore(total_data[c].values)


    #PCA_value
    coord = total_origin_data[['lat', 'long']]
    pca = PCA(n_components=2)
    pca.fit(coord)
    coord_pca = pca.transform(coord)
    total_data['coord_pca1'] = coord_pca[:, 0]
    total_data['coord_pca2'] = coord_pca[:, 1]

    #create zip code value
    search = SearchEngine(simple_zipcode=True)
    total_data['major_city'] = [search.by_zipcode(str(i)).major_city for i in total_data['zipcode']]
    total_data.loc[(total_data['major_city'] =='Medina') | (total_data['major_city'] == 'Mercer Island'), 'major_city'] = 1
    total_data.loc[total_data['major_city'] != 1, 'major_city'] = 0
    total_data['major_city'] = total_data['major_city'].astype('float')


    droplist = ['long','yr_built','sqft_lot15', 'sqft_living15', 'waterfront', 'zipcode','cluster',
                'condition','yr_renovated', 'view', 'total_room', 'floors']
    total_data = total_data.drop(droplist, axis=1)


    ##final_data
    final_data_x, final_data_y = total_data.drop(['price'], axis=1), total_data[['price']]
    final_data_y = np.log1p(final_data_y)

    return final_data_x, final_data_y


def clip_outlier(origin_data, column):
    total_data_np = origin_data[column]
    quan_25 = np.percentile(total_data_np.values, 25)
    quan_75 = np.percentile(total_data_np.values, 75)

    iqr = quan_75 - quan_25
    iqr = iqr * 1.5
    lowest = quan_25 - iqr
    highest = quan_75 + iqr
    # lowest_index = np.array(total_data_np[(total_data_np < lowest)].index)
    # highest_index = np.array(total_data_np[(total_data_np > highest)].index)

    origin_data.loc[origin_data[column] < lowest, column] = lowest
    origin_data.loc[origin_data[column] > highest, column] = highest

    return origin_data


class simple_regression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(simple_regression, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.fc1 = nn.Sequential(
            nn.Linear(self.input_dim, 64),
            nn.LeakyReLU(),

            nn.Linear(64, 64),
            nn.LeakyReLU(),

            nn.Linear(64, 64),
            nn.LeakyReLU(),

            nn.Linear(64, self.output_dim)
        )

        self.cuda()

    def forward(self, x):
        x = self.fc1(x.cuda())
        return x.cpu()


class TensorData(Dataset):

    def __init__(self, x_data, y_data):
        self.x_data = torch.FloatTensor(x_data)
        self.y_data = torch.FloatTensor(y_data)
        self.len = self.y_data.shape[0]

    def __getitem__(self, index):

        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


def plot(loss):
    plt.plot(loss)
    plt.title('loss')
    plt.xlabel('epoch')
    plt.show()