
from sklearn.model_selection import ShuffleSplit
from model_cross_validation import *
import matplotlib.pyplot as plt
plt.rcParams["axes.grid"] = False
from sklearn import datasets
import torch
from torch import nn, optim
import numpy as np
from matplotlib import pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from sklearn.datasets import make_swiss_roll
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import KBinsDiscretizer
import warnings
warnings.filterwarnings("ignore")
from pytorch_complex_tensor import ComplexTensor




def main():

    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    dataset, colors = datasets.make_swiss_roll(8000, noise=0.0, random_state=None)
    scale_normalize = MinMaxScaler(feature_range=(0, 1))
    scale_normalize.fit(dataset)
    X_data_norm = scale_normalize.transform(dataset)


    p_test = 0.2
    n_sims, fold = 1, 2
    n_alg = 2
    ss = ShuffleSplit(n_splits=n_sims, test_size=p_test)
    md = [[] for n in range(n_alg)]
    MSE_test = np.zeros((n_sims, n_alg))

    for it, ind in enumerate(ss.split(X_data_norm)):

        print(f'Simulation number : {it}')
        ind_tr, ind_te = ind
        s_tr, s_te = len(ind_tr), len(ind_te)

        # Training phase
        X_norm_train = X_data_norm[ind_tr]
        X_norm_test = X_data_norm[ind_te]

        # Parameters!!! Choose carefully
        n_coef = [11]
        F = [10]

        alpha = [1e-2] # Regularization
        lr = [1e-1] # Learning Rate

        X_samples = cda_regression_sgd(X_norm_train, X_norm_test, scale_normalize, n_coef, alpha, F, lr, fold, b_size=500, max_iter=50)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for tick in ax.get_xticklabels():
            tick.set_rotation(45)
        #ax.scatter([data[i][0] for i in range(len(X))], [X[i][1] for i in range(len(X))],[X[i][2] for i in range(len(X))], s=.6,c='k', marker='.')
        ax.scatter([X_norm_train[i][0] for i in range(int(len(X_norm_train)))], [X_norm_train[i][1] for i in range(int(len(X_norm_train)))],
                   [X_norm_train[i][2] for i in range(int(len(X_norm_train)))],s=20, c='k', marker='.')
        ax.scatter([X_samples[i][0] for i in range(len(X_samples))], [X_samples[i][1] for i in range(len(X_samples))],
                   [X_samples[i][2] for i in range(len(X_samples))], s=10, c='dodgerblue', marker='.')
        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')
        ax.set_zlabel('$x_3$')
        ax.axis('off')

        plt.show()




if __name__ == '__main__':
            main()
