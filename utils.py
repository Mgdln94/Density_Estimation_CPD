import numpy as np
import math
#import matplotlib.pyplot as plt
from utils import *


def plot_tensor_weights(factors):
    N = len(factors)
    F = factors[0].shape[1]
    scaling = np.ones((1, F))
    for n in range(N):
        scaling = scaling*np.linalg.norm(factors[n], axis=0)
    scaling = np.sort(scaling)
    fig = plt.figure(1, figsize=[15, 6])
    plt.clf()
    plt.semilogy(scaling[0][::-1]/np.sum(scaling[0]))
    plt.grid()
    plt.pause(0.1)

def visualize_data(X):
    N = X.shape[1]
    fig = plt.figure(1, figsize=[15, 6])
    for n in range(N):
        fig.add_subplot(1, N, n + 1)
        plt.hist(X[:, n], bins=20)
    plt.show()


def load_dataset(d_name):

    if d_name == 'QSAR':
        data = np.genfromtxt('./Datasets/'+d_name+'.data', delimiter=',')
        targets = data[:, -1]
        data = data[:, :-1]

    elif d_name == 'Energy_efficiency_1':
        data = np.genfromtxt('./Datasets/'+d_name+'.data', delimiter=',')
        targets = data[:, -1]
        data = data[:, :-1]

    elif d_name == 'Energy_efficiency_2':
        data = np.genfromtxt('./Datasets/'+d_name+'.data', delimiter=',')
        targets = data[:, -1]
        data = data[:, :-1]

    elif d_name == 'Airfoil':
        data = np.genfromtxt('./Datasets/'+d_name+'.data', delimiter='\t')
        targets = data[:, -1]
        data = data[:, :-1]

    elif d_name == 'Skillcraft':
        data = np.genfromtxt('./Datasets/'+d_name+'.data', delimiter=',')
        targets = data[:, -1]
        data = data[:, :-1]

    elif d_name == 'Abalone':
        data = np.genfromtxt('./Datasets/'+d_name+'.data', delimiter=',')
        targets = data[:, -1]
        data = data[:, 1:-1] # The first one is discrete

    elif d_name == 'Combined_Cycle_Power_Plant':
        data = np.genfromtxt('./Datasets/'+d_name+'.data', delimiter=',')
        targets = data[:, -1]
        data = data[:, :-1]

    elif d_name == 'Physicochemical':
        data = np.genfromtxt('./Datasets/'+d_name+'.data', delimiter=',')
        targets = data[:, -1]
        data = data[:, :-1]

    elif d_name == 'Superconduct':
        data = np.genfromtxt('./Datasets/'+d_name+'.data', delimiter=',')
        targets = data[:, -1]
        data = data[:, :-1]

    return data, targets