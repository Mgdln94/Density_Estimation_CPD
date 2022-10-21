

### Magdalena (Magda) Amiridi, UVA 2020
### Code for my paper https://arxiv.org/abs/2008.12315
### Nonparametric Multivariate Density Estimation: A Low-Rank Characteristic Function Approach

import htf_sgd
import itertools
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
from sklearn.model_selection import KFold
from utils import *
from pytorch_complex_tensor import ComplexTensor
from scipy.integrate import trapz
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer
import warnings
warnings.filterwarnings("ignore")

# Get conditional densities in each dimension d
def get_conditionals_per_latent(md,n, h, t):

    bbx = np.array(
        np.exp(np.outer(-1j * 2 * math.pi * np.arange((-md.coef_size - 1) / 2 + 1, (md.coef_size - 1) / 2 + 1), t)),
        dtype=np.complex64)
    bb1 = torch.complex(torch.tensor(bbx.real), torch.tensor(bbx.imag))
    aug_fac = torch.complex(torch.cat((md.factors[n].real, torch.ones(1, md.F), torch.tensor(md.factors[n].real).__reversed__()), dim=0),
                                    torch.cat((md.factors[n].imag, torch.zeros(1, md.F), torch.tensor(md.factors[n].imag.neg()).__reversed__()), dim=0))

    res = ((bb1.t().mm((aug_fac))).real)
    res[res < 0] = 0  # Note here: res can take negative values, take max{0,red} -> res[res < 0] = 0, Ensure positivity
    res_sc = (bb1.t().mm((aug_fac))).real
    res_sc[res_sc < 0] = 0
    res_sc = res_sc.data
    for col in range(res_sc.shape[1]):
        c = 1 / trapz(res_sc[:, col],t)
        res.data[:, col] = c * (res.data[:, col])  # Normalize conditional density, Ensure integration to 1
    res = res.detach().numpy()
    return res[:,h]

# Rejection sampling
def rejection_sampler(pdf, t, num_samples=1, xmin=0, xmax=1): # Code taken from https://gist.github.com/rsnemmen/d1c4322d2bc3d6e36be8
    pmin = 0.
    pmax = pdf.max()
    # Counters
    naccept = 0
    ntrial = 0
    # Keeps generating numbers until we achieve the desired n
    ran = []  # output list of random numbers
    while naccept < num_samples:
        x = np.random.uniform(xmin, xmax)  # x'
        x_indx = (np.abs(np.array(t) - x)).argmin()
        y = np.random.uniform(pmin, pmax)  # y'
        if y < pdf[x_indx]:
            ran.append(x)
            naccept = naccept + 1
        ntrial = ntrial + 1
    return np.asarray(ran)

def sample(md, num_samples, X_train):

    facs = [fact.detach().numpy() for fact in md.factors]
    rand_lamda_index =  np.random.choice(md.F, num_samples, replace=True, p= np.squeeze((facs[-1])/(facs[-1]).sum()))
    samples = np.ones((num_samples, md.N))
    for i in range(num_samples):
        for n in range(md.N): #rejection_sampler(get_conditionals_per_latent(md,n,t)[:,h], t, num_samples)
            t = sorted(np.array(list(set(([float(X_train[m][n]) for m in range(X_train.shape[0])])))))
            if (len(t) > 50):
                discretizer = KBinsDiscretizer(n_bins=50, encode='ordinal', strategy='kmeans')
                discretizer.fit((np.array(t)).reshape((len(t), 1)))
                disc_data = discretizer.transform((np.array(t)).reshape((len(t), 1)))
                disc_data = discretizer.inverse_transform(disc_data)
                t = disc_data.reshape((len(disc_data),))
            samples[i,n] = rejection_sampler(get_conditionals_per_latent(md,n,rand_lamda_index[i],t), t) #continuous case
    return samples

def get_basis(md, x):
    M = x.shape[0]
    basis = []
    for n in range(md.N):
        basis.append(np.zeros((md.coef_size, M)))
        basis[n] = np.exp(
            np.outer(-1j * 2 * math.pi * np.arange((-md.coef_size - 1) / 2 + 1, (md.coef_size - 1) / 2 + 1),
                     x[:, n]))
    return basis

def get_likelihood(md, x):
    basis = get_basis(md, x)
    basis = torch.complex(torch.tensor(np.array(basis, dtype=np.complex64).real),
                                     torch.tensor(np.array(basis, dtype=np.complex64).imag))

    aug_fac = torch.complex(
        torch.cat((md.factors[0].real, torch.ones(1, md.F), md.factors[0].real.__reversed__()), dim=0),
        torch.cat((md.factors[0].imag, torch.zeros(1, md.F), md.factors[0].imag.neg().__reversed__()), dim=0))

    c = np.ones((md.N, md.F))
    res = ((basis[0]).t().mm((aug_fac))).real
    res[res <= 0] = 0.000001
    res = torch.Tensor(c[0, :]) * res
    res = torch.log(res)
    for n in range(1, md.N):
        aug_fac = torch.complex(
            torch.cat((md.factors[n].real, torch.ones(1, md.F), md.factors[n].real.__reversed__()), dim=0),
            torch.cat((md.factors[n].imag, torch.zeros(1, md.F), md.factors[n].imag.neg().__reversed__()),
                      dim=0))
        tmp = ((basis[n]).t().mm((aug_fac))).real
        tmp[tmp <= 0] = 0.000001
        tmp = torch.log(tmp)
        tmp = torch.Tensor(c[n, :]) * tmp  # Normalize conditional density
        res = res + tmp

    res = torch.exp(res)
    fnl_results = torch.sum((res.mm(torch.diagflat(md.factors[-1]))), dim=1)
    return fnl_results

def cda_regression_sgd(X, X_test,scale_normalize, n_coef, alpha, F, lr, fold, b_size=512, max_iter=1000, tol=1e-2):
    print('Training Characterstic Function Based Density Estimation - HTF SGD')

    coef_s, a, f, lr = n_coef[0], alpha[0], F[0], lr[0]
    md = htf_sgd.htf_sgd(coef_s, a, f, lr, b_size, max_iter, tol)
    md.fit(X, X_test)

    # Likelihood eval

    x = X[340, :].reshape(-1,1).T
    gt = x

    # Let us calculate the ML estimate of X1
    n=0
    likelihood = []
    alphabet = sorted(X[:,n])
    for i in alphabet:
        x[n][0] = i
        likelihood.append(get_likelihood(md, x))
    max_index = likelihood.index(max(likelihood))
    x[n][0] = alphabet[max_index]
    print(gt,x)

    # Sample from the distribution --> Latent variable NB H->X
    num_samples = 5000
    X_samples = sample(md, num_samples, X)

    return X_samples # Return synthetic samples
