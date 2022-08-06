import datasets
import numpy as np
import pandas as pd
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import uilts

# Device configuration
device = "cuda" if torch.cuda.is_available() else "cpu"


# Hyper parameters
num_epochs = 50
batch_size = 50
num_classes = 100
learning_rate = 0.0001

## data_load
dataset_odn = datasets.C100Dataset('../dataset/data/cifar100.csv')
[data_train_x, data_train_y, data_test_x, data_test_y] = dataset_odn.getDataset()


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

model = SimpleCNN().to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

##  data_sampling with batch & Train the model
total_train_step = int( len(data_train_x) / batch_size)
total_test_step = int( len(data_test_x) / batch_size)

train_correct = []
test_correct = []

for epoch in range(num_epochs):
    correct = 0
    total = 0

    for i in range(total_train_step):
        shuffled = np.random.permutation(data_train_x.shape[0])
        data_train_x = data_train_x[shuffled, :]
        data_train_y = data_train_y[shuffled]

        images = data_train_x[i * batch_size: (i + 1) * batch_size]
        images = torch.from_numpy(images).permute(0, 3, 1, 2).to(device, dtype=torch.float)

        labels = data_train_y[i * batch_size: (i + 1) * batch_size]
        labels = torch.from_numpy(labels).to(device, dtype=torch.long)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, argmax = torch.max(outputs, 1)
        accuracy = (labels == argmax).float().mean()

        total += labels.size(0)
        correct += (argmax == labels).sum().item()

        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Train Accuracy: {:.2f}%'.format(epoch + 1, num_epochs, i + 1, total_train_step, loss.item(), accuracy.item()*100))

    print('Epoch [{}/{}], Train Accuracy of the model on the 50000 Train images: {} %'.format(epoch + 1, num_epochs, 100 * correct / total))
    train_correct.append(100*correct/total)

    # Test the model
    model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)

    with torch.no_grad():
        correct = 0
        total = 0
        predicted_list = []

        for i in range(total_test_step): ##test data가 data_size만큼 뽑히게 만들어주어야한다.

            test_images = data_test_x[i * batch_size: (i + 1) * batch_size]
            test_images = torch.from_numpy(test_images).permute(0, 3, 1, 2).to(device, dtype=torch.float)
            test_images = torch.reshape(test_images, [-1, 3, 32, 32])

            labels = data_test_y[i * batch_size: (i + 1) * batch_size]
            labels = torch.from_numpy(labels).to(device, dtype=torch.long)

            outputs = model(test_images)

            _, predicted = torch.max(outputs, 1)
            accuracy = (labels == predicted).float().mean()

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            predicted = predicted.detach().cpu()
            predicted_list.append(predicted.numpy())

            if (i + 1) % 20 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Test Accuracy: {:.2f}%'.format(epoch + 1, num_epochs, i + 1,total_test_step, accuracy.item() * 100))

        test_correct.append(100 * correct / total)
        print('Epoch [{}/{}], Test Accuracy of the model on the 10000 test images: {} %'.format(epoch + 1, num_epochs,100 * correct / total))

        predicted_list = np.array(predicted_list).flatten()

        ##replace origin_name
        data = pd.read_csv('../dataset/data/cifar100.csv', names=['path', 'class'], header=None)
        class_name = np.unique(data['class'])
        origin_class = {key: value for key, value in enumerate(class_name)}

        ### To submit kaggle
        test_list = pd.read_csv('../dataset/kaggle.csv')
        kaggle_submission = pd.DataFrame({'Id': test_list['id'], 'Category': predicted_list})
        kaggle_submission = kaggle_submission.replace(origin_class)
        kaggle_submission.to_csv('../dataset/kaggle_submission.csv')

train_correct = np.array(train_correct).flatten()
test_correct = np.array(test_correct).flatten()

uilts.plot_results(num_epochs, train_correct / total, test_correct / total)

# Save the model checkpoint
torch.save(model.state_dict(), '../model.ckpt')

