from matplotlib import pyplot as plt
import numpy as np


def plot_results(epochs, train_correct, test_correct):
    epoch_len = np.arange(epochs)
    plt.plot(epoch_len, train_correct, label='training')
    plt.plot(epoch_len, test_correct, label='testing')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('accuracy plot')
    plt.draw()

    plt.savefig( '../report/' + 'total_accuracy' + '_graph.png')

