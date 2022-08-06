from matplotlib import pyplot as plt
import numpy as np
import torch
import os
import pandas as pd
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
import torchvision.transforms as transforms

from random import seed
from random import randrange
import copy
from PIL import Image

# Device configuration
device = "cuda" if torch.cuda.is_available() else "cpu"

## set the model
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, 100)

    def forward(self, x):
        x = self.conv(x)
        x = x.reshape(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class dataloader:
    def __init__(self,x_data, y_data, i, batch_size, mode='train'):
        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomRotation(5),
                transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop(32, scale=(0.96, 1.0), ratio=(0.95, 1.05)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize([32, 32]),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),

            'test': transforms.Compose([
                transforms.Resize([32, 32]),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        }

        image_list = torch.empty(batch_size, 3, 32, 32)

        for j, index in zip(range(batch_size) ,np.arange(i * batch_size, (i + 1) * batch_size)):
            image = x_data[index]
            image = Image.fromarray(image)
            image = data_transforms[mode](image)
            image_list[j] = image

        labels = y_data[i * batch_size: (i + 1) * batch_size]
        labels = torch.from_numpy(labels)

        self.images = image_list
        self.labels = labels

    def getload(self):
        return self.images, self.labels


def plot_results(epochs, train_correct, valid_correct):
    epoch_len = np.arange(epochs)
    plt.plot(epoch_len, train_correct, label='training')
    plt.plot(epoch_len, valid_correct, label='validation')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('accuracy plot')
    plt.draw()
    plt.savefig( '../report/' + 'total_accuracy' + '_graph.png')


def shuffle_dataset(x_data, y_data):
    shuffled = np.random.permutation(x_data.shape[0])
    return x_data[shuffled,:], y_data[shuffled]


def split_train_val(x_data, y_data):
    validation_rate = 0.2
    validation_num = int(x_data.shape[0] * validation_rate)

    x_train = x_data[validation_num:]
    y_train = y_data[validation_num:]
    x_valid = x_data[:validation_num]
    y_valid = y_data[:validation_num]

    return x_train, y_train, x_valid, y_valid

def cross_validation_split(x_data, y_data, num_folds=5):

    # data preparation
    x_data_copy = list(copy.deepcopy(x_data))
    y_data_copy = list(copy.deepcopy(y_data))
    x_dataset = []
    y_dataset = []

    # fold setting
    seed(514)
    fold_size = int(len(x_data) / num_folds)  # 전체 데이터에서 5분의1

    for _ in range(num_folds):
        fold_x = []
        fold_y = []
        while len(fold_x) < fold_size:
            fold_index = randrange((len(x_data_copy)))
            fold_x.append(x_data_copy.pop(fold_index))
            fold_y.append(y_data_copy.pop(fold_index))

        x_dataset.append(fold_x)
        y_dataset.append(fold_y)

    return x_dataset, y_dataset



def validation(epoch, model, data_valid_x, data_valid_y, criterion, device, batch_size):
    print('Start validation #{}'.format(epoch) )
    model.eval()
    with torch.no_grad():
        total = 0
        correct = 0
        total_loss = 0
        cnt = 0
        total_valid_step = int(len(data_valid_x)/ batch_size)
        for i in range(total_valid_step):
            imgs, labels = dataloader(data_valid_x, data_valid_y, i, batch_size, mode='val').getload()
            imgs = imgs.to(device, dtype=torch.float)
            labels = labels.to(device, dtype=torch.long)

            #imgs = data_valid_x[i * batch_size: (i + 1) * batch_size]
            #imgs = torch.from_numpy(imgs).permute(0, 3, 1, 2).to(device, dtype=torch.float)
            #imgs = torch.from_numpy(imgs).to(device, dtype=torch.float)

            #labels = data_valid_y[i * batch_size: (i + 1) * batch_size]
            #labels = torch.from_numpy(labels).to(device, dtype=torch.long)

            outputs = model(imgs)
            loss = criterion(outputs, labels)

            total += imgs.size(0)
            _, argmax = torch.max(outputs, 1)
            correct += (labels == argmax).sum().item()
            total_loss += loss
            cnt += 1
        avrg_loss = total_loss / cnt
        valid_correct = correct / total * 100
        print('Validation #{}  Accuracy: {:.2f}%  Average Loss: {:.4f}'.format(epoch, correct / total * 100, avrg_loss))
    model.train()
    return avrg_loss, valid_correct

def save_model(model, saved_dir, file_name='best_model.pt'):
    os.makedirs(saved_dir, exist_ok=True)
    check_point = {
        'net': model.state_dict()
    }
    output_path = os.path.join(saved_dir, file_name)
    torch.save(check_point, output_path)


