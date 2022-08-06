import datasets
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import utils
import pickle

#config
saved_dir = '../saved/CNN_pre'

# Device configuration
device = "cuda" if torch.cuda.is_available() else "cpu"

# Hyper parameters
train_epochs = 10
batch_size = 128
best_loss = 999999999
val_every = 1
learning_rate = 0.0001
total_epochs = 0


# Model & Loss & optimizer
model = utils.SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# cross validation & training

train_correct_list = []
valid_correct_list = []

print('Start training..')
for epoch in range(train_epochs):

    ## data_load
    dataset_odn = datasets.C100Dataset('../dataset/cifar100_nl/data/p_cifar.csv')
    [data_train_x, data_train_y, data_test_x, data_test_y] = dataset_odn.getDataset()
    x_dataset, y_dataset = utils.cross_validation_split(data_train_x, data_train_y)

    n_iter = 0

    for i in range(len(x_dataset)):
        train_x = list(x_dataset)
        train_x.pop(i)
        train_x = np.asarray(train_x).reshape(-1, 32, 32, 3)

        train_y = list(y_dataset)
        train_y.pop(i)
        train_y = np.asarray(train_y).reshape(-1)

        valid_x = np.asarray(x_dataset[i]).reshape(-1, 32, 32, 3)
        valid_y = np.asarray(y_dataset[i])

        n_iter += 1
        total_epochs += 1
        total_train_step = int(len(train_x) / batch_size)
        train_total = 0
        correct = 0

        ### training
        for i in range(total_train_step):
            train_x, train_y = utils.shuffle_dataset(train_x, train_y)
            train_images, train_labels = utils.dataloader(train_x, train_y, i, batch_size, mode='train').getload()
            train_images = train_images.to(device, dtype=torch.float)
            train_labels = train_labels.to(device, dtype=torch.long)

            # Forward pass
            outputs = model(train_images)
            loss = criterion(outputs, train_labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, argmax = torch.max(outputs, 1)
            accuracy = (train_labels == argmax).float().mean()

            train_total += train_labels.size(0)
            correct += (argmax == train_labels).sum().item()

            if (i + 1) % 100 == 0:
                print(
                    'Epoch [{}/{}], Fold [{}/{}], Step [{}/{}], Loss: {:.4f}, Train Accuracy: {:.2f}%'.format(total_epochs, train_epochs*5, n_iter, 5,
                                                                                                              i + 1,
                                                                                                              total_train_step,
                                                                                                              loss.item(),
                                                                                                              accuracy.item() * 100))
        train_correct_list.append( 100 * correct / train_total)

        if total_epochs % val_every == 0:
            avrg_loss, valid_correct = utils.validation(total_epochs, model, valid_x, valid_y, criterion,
                                                                          device, batch_size)
            if avrg_loss < best_loss:
                print('Best performance at epoch: {}'.format(total_epochs))
                print('Save model in', saved_dir)
                best_loss = avrg_loss
                utils.save_model(model, saved_dir)
        valid_correct_list.append(valid_correct)

train_correct_list = np.array(train_correct_list).flatten()
valid_correct_list = np.array(valid_correct_list).flatten()


with open('../saved/CNN/correct_list/train_correct.pickle', 'wb') as f:
    pickle.dump(train_correct_list, f)

with open('../saved/CNN/correct_list/valid_correct.pickle', 'wb') as f:
    pickle.dump(valid_correct_list, f)


utils.plot_results(total_epochs, train_correct_list , valid_correct_list)
