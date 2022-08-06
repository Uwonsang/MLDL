import numpy as np
import pandas as pd
from PIL import Image

class C100Dataset:
    """
    X is a feature vector
    Y is the predictor variable
    """
    train_x = None  # X (data) of training set.
    train_y = None  # Y (label) of training set.
    test_x = None # X (data) of test set.
    test_y = None # Y (label) of test set.

    def __init__(self, filename):
        ## read the csv for dataset (cifar100.csv, cifar100_lt.csv or cifar100_nl.csv), 
        # 
        # Format:
        #   image file path,classname
        
        ### TODO: Read the csv file and make the training and testing set
        ## YOUR CODE HERE

        data = pd.read_csv(filename, names=['path', 'class'], header=None)
        class_name = np.unique(data['class'])
        one_hot_class = {value: key for key, value in enumerate(class_name)}
        data = data.replace(one_hot_class)

        data_img = []

        for filename in data['path']:
            img_data = Image.open('../dataset/' + filename)
            img_array = np.asarray(img_data) / 255.0
            data_img.append(img_array)

        data_img = np.asarray(data_img)

        ### TODO: assign each dataset
        self.train_x = data_img[:50000] ### TODO: YOUR CODE HERE
        self.train_y = np.asarray(data['class'][:50000]) ### TODO: YOUR CODE HERE
        self.test_x = data_img[50000:] ### TODO: YOUR CODE HERE
        self.test_y = np.asarray(data['class'][50000:]) ### TODO: YOUR CODE HERE

    def getDataset(self):
        return [self.train_x, self.train_y, self.test_x, self.test_y]


if __name__ == '__main__':
    dataset_odn = C100Dataset('../dataset/data/cifar100.csv')
    [data_odn_tr_x, data_odn_tr_y, data_odn_ts_x, data_odn_ts_y] = dataset_odn.getDataset()