import torch
import os


def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum / y_test.shape[0]
    acc = torch.round(acc * 100)

    return acc


def load_pretrained_embedding():
    weights = []

    with open(os.path.join('.', 'preprocessed_data', '10k_embedding.txt')) as f:
        embedding = f.readlines()

        for line in embedding:
            rows = line.split()[1:]
            rows = list(map(float, rows))
            weights.append(rows)

    weights.append([-1.] * 100)
    weights.append([0.] * 100)

    weights = torch.FloatTensor(weights)

    return weights






