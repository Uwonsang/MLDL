import datasets
import pandas as pd
import numpy as np
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import utils
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler


##parameter
num_epochs = 100
learning_rate = 0.001

# Device configuration
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)


def main():
    price_dataset = datasets.PriceDataset()

    '''load train'''
    [train_x, train_y, valid_x, valid_y] = price_dataset.getDataset()

    '''load test'''
    test = pd.read_csv('data/price_data_ts.csv')
    test['date'] = test['date'].apply(lambda x: str(x[:6])).astype('float')
    test_x, test_y = test.drop(['price'], axis=1), test[['price']]

    total_x = pd.concat((train_x, valid_x))
    total_y = pd.concat((train_y, valid_y))

    '''make cluster'''
    cluster_x = pd.concat((total_x, test_x))
    location = pd.concat((cluster_x['lat'], cluster_x['long']), axis=1)
    kmeans = KMeans(n_clusters=5).fit(location)
    cluster_x['cluster'] = pd.DataFrame(kmeans.labels_)
    total_x['cluster'], test_x['cluster'] = cluster_x['cluster'][:len(total_x)], cluster_x['cluster'][len(total_x):]


    '''preprocessing the data'''
    pre_total_x, pre_total_y = utils.preprocessing(total_x, total_y)

    train_index = int(pre_total_x.shape[0] * 0.75)
    pre_train_x, pre_valid_x = pre_total_x[0:train_index], pre_total_x[train_index:]
    pre_train_y, pre_valid_y = pre_total_y[0:train_index], pre_total_y[train_index:]


    '''data_to_batch'''
    train_set = utils.TensorData(pre_train_x.to_numpy(), pre_train_y.to_numpy())
    train_loader = DataLoader(train_set, batch_size = 32, shuffle=True)

    valid_set = utils.TensorData(pre_valid_x.to_numpy(), pre_valid_y.to_numpy())
    valid_loader = DataLoader(valid_set, batch_size= 32, shuffle=False)

    '''load model'''
    model = utils.simple_regression(pre_train_x.shape[1], pre_train_y.shape[1]).to(device)

    '''set Loss and optimizer'''
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001,  weight_decay=1e-7)

    total_train_loss = []

    for epoch in range(num_epochs):

        training_loss = 0.0
        for i, train_batch in enumerate(train_loader):
            train_x_batch, train_y_batch = train_batch

            optimizer.zero_grad()

            outputs = model(train_x_batch)
            loss = criterion(outputs, train_y_batch)
            loss.backward()
            optimizer.step()

            training_loss += loss.item()

            if epoch % 10 == 0:
                print(f"Train_Epoch: {epoch} | MSE loss: {loss}")

        total_train_loss.append(training_loss / len(train_loader))

    utils.plot(total_train_loss)

    # Validate the model
    model.eval()
    with torch.no_grad():

        valid_loss = []
        true_list = []
        pred_list = []
        x_list = []
        x_column_list = pre_train_x.columns.tolist()

        for i, valid_batch in enumerate(valid_loader):
            valid_x_batch, valid_y_batch = valid_batch

            outputs = model(valid_x_batch)
            loss = torch.sqrt(criterion(np.exp(outputs), np.exp(valid_y_batch)) + 1e-7)

            valid_loss.append(loss.item())
            # print(f"valid_num: {i} | RMSE loss: {loss}\n ")

            x_list.append(valid_x_batch.tolist())
            true_list.append(np.exp(valid_y_batch).tolist())
            pred_list.append(np.exp(outputs).tolist())

        print(f'TotalRMSE loss: {np.mean(valid_loss)}\n')
        pred = pd.DataFrame(sum(pred_list, []), columns= ['pred'])
        true = pd.DataFrame(sum(true_list, []), columns= ['true'])
        x = pd.DataFrame(sum(x_list, []), columns = x_column_list)
        loss = pd.DataFrame(abs(pred['pred'] - true['true']), columns=['loss'])

        valid_data = pd.concat([x, pred, true, loss], axis=1)
        top_10 = valid_data.sort_values(by=['loss'])[:10]
        low_10 = valid_data.sort_values(by=['loss'])[4313:]
        

        with open(os.path.join('.', 'data', 'valid_data.csv'), 'w') as f:
            f.write(valid_data.to_csv(index=False))

    ## test model
    test_x, _ = utils.preprocessing(test_x, test_y, type='test')
    test_x = torch.FloatTensor(test_x.to_numpy())


    predict_y = model(test_x)
    predict_y = np.expm1(predict_y.cpu().detach().numpy().astype(np.float32))
    predict_y = predict_y.reshape(-1)


    origin_test = pd.read_csv('data/price_data_ts.csv')
    origin_test['id2'] = ['%010d' % (row['id']) + str(row['date']) for _, row in origin_test.iterrows()]

    pred_df = pd.DataFrame({
        'id': origin_test['id2'].values,
        'price': predict_y
    })

    with open(os.path.join('.', 'data', 'result.csv'), 'w') as f:
        f.write(pred_df.to_csv(index=False))


if __name__ == '__main__':
    main()