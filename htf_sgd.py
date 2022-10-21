import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
from collections import deque
from utils import *
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import KBinsDiscretizer
import warnings
warnings.filterwarnings("ignore")
from scipy.integrate import trapz


def projection_simplex_Eucl(y):
    D = len(y)
    u = np.sort(y)[::-1]
    arg_u = np.argsort(y)[::-1]

    rho = 0
    for j in range(D):
        if u[j] + 1 / (j + 1) * (1 - np.sum(u[0:j + 1])) > 0:
            rho = j + 1
    lambda_val = 1.0 / rho * (1 - np.sum(u[0:rho]))

    x = np.zeros([D])
    for i in range(D):
        x[i] = max(y[i] + lambda_val, 0)
    return np.float32(x)


def train(model, optimiser, train_dataloader):
    model.train()
    for idx, data in enumerate(train_dataloader):
        x = data
        optimiser.zero_grad()
        y_hat = model(x, idx, 'train')
        loss = torch.mean(-torch.log(y_hat[y_hat > 0]))
        loss.backward()
        optimiser.step()
        model.factors[-1].data = torch.tensor(projection_simplex_Eucl((((model.factors[-1]).clone()).detach().numpy())).reshape(model.F, 1))

def validate(model, val_dataloader, M):
    model.eval()
    validation_loss = 0
    with torch.no_grad():
        for idx, data in enumerate(val_dataloader):
            x = data
            y_hat = model(x, idx, 'val')
            validation_loss += torch.mean(-torch.log(y_hat[y_hat > 0]))
    return validation_loss / M


class csid_net(nn.Module):
    def __init__(self, x_train, x_val, F, shape, b_size):

        super(csid_net, self).__init__()

        self.F, self.shape = F, shape  # list of K = [K1 K2 ... KN]
        self.ndims = len(self.shape)  # N

        init_params = np.array(self.init_factors(), dtype=np.complex64)
        factors_init = torch.complex(torch.tensor(init_params.real),torch.tensor(init_params.imag))

        self.basis_train = self.get_basis(x_train)
        self.basis_val = self.get_basis(x_val)

        self.basis_train = torch.complex(torch.tensor(np.array(self.basis_train, dtype=np.complex64).real), torch.tensor(np.array(self.basis_train, dtype=np.complex64).imag))
        self.basis_val = torch.complex(torch.tensor(np.array(self.basis_val, dtype=np.complex64).real), torch.tensor(np.array(self.basis_val, dtype=np.complex64).imag))
        self.factors = nn.ParameterList(([nn.Parameter(factors_init[n]) for n in range(self.ndims)]))
        lst = []
        [lst.append(random.uniform(0, 1)) for i in range(self.F)]
        f_vec = f_vec = torch.Tensor(((1 / (self.F) * np.ones(self.F)).reshape(self.F,
                                                                               1)))  # ComplexTensor((1 / (self.F) * np.ones(self.F)).reshape(self.F,1))
        self.factors.append(nn.Parameter(f_vec))
        self.idx_train = []

        for i in range(math.ceil(x_train.shape[0] / b_size)):
            l = i * b_size
            r = l + b_size if l + b_size < x_train.shape[0] else x_train.shape[0]
            self.idx_train.append(range(l, r))

        self.idx_val = []
        for i in range(math.ceil(x_val.shape[0] / b_size)):
            l = i * b_size
            r = l + b_size if l + b_size < x_val.shape[0] else x_val.shape[0]
            self.idx_val.append(range(l, r))

    def init_factors(self):
        factors = []
        for n in range(self.ndims):
            factors.append(np.random.rand(int((self.shape[n] - 1) / 2), self.F) + 0.1j * np.random.rand(
                int((self.shape[n] - 1) / 2), self.F))
        return factors

    def get_basis(self, x):
        M = x.shape[0]
        basis = []
        for n in range(self.ndims):
            basis.append(np.zeros((self.shape[0], M)))
            basis[n] = np.exp(
                np.outer(-1j * 2 * math.pi * np.arange((-self.shape[0] - 1) / 2 + 1, (self.shape[0] - 1) / 2 + 1),
                         x[:, n].numpy()))
        return basis

    def get_conditionals_per_latent(self, n, x):

        # If the alphabet is relatively small, get all possible values
        t = np.array(list(set(([float(x[0].data[m][n]) for m in range(x[0].shape[0])]))))
        (t.reshape((len(t),))).sort()

        # If the alphabet is too large, discretize into 200 bins using Kmeans
        if (len(t) > 50):
            discretizer = KBinsDiscretizer(n_bins=50, encode='ordinal', strategy='kmeans')
            discretizer.fit(t.reshape((len(t), 1)))
            disc_data = discretizer.transform(t.reshape((len(t), 1)))
            disc_data = discretizer.inverse_transform(disc_data)
            t = disc_data.reshape((len(disc_data),))


        bbx = np.array(np.exp(np.outer(-1j * 2 * math.pi * np.arange((-self.shape[0] - 1) / 2 + 1, (self.shape[0] - 1) / 2 + 1), t)),
                                     dtype=np.complex64)
        bb1 = torch.complex(torch.tensor(bbx.real), torch.tensor(bbx.imag))

        aug_fac = torch.complex(torch.cat((self.factors[n].real, torch.ones(1, self.F), self.factors[n].real.__reversed__()), dim=0),
                                    torch.cat((self.factors[n].imag, torch.zeros(1, self.F), self.factors[n].imag.neg().__reversed__()), dim=0))


        res = ((bb1.t().mm((aug_fac))).real)
        res[
            res < 0] = 0.000001  # Note here: res can take negative values, take max{0,red} -> res[res < 0] = 0, Ensure positivity
        res_sc = (bb1.t().mm((aug_fac))).real
        res_sc[res_sc < 0] = 0.000001
        res_sc = res_sc.data

        c = np.zeros(self.F)
        for col in range(res_sc.shape[1]):
            c[col] = 1 / trapz(res_sc[:, col], t)
        return c

    def forward(self, x, idx, state):
        if state == 'train':

            aug_fac = torch.complex(torch.cat((self.factors[0].real, torch.ones(1, self.F), self.factors[0].real.__reversed__()), dim=0),
                                    torch.cat((self.factors[0].imag, torch.zeros(1, self.F), self.factors[0].imag.neg().__reversed__()), dim=0))

            # Ensure sum to 1 conditionals
            c = np.ones((self.ndims, self.F))
            for j in range(self.ndims):
                c[j] = self.get_conditionals_per_latent(j, x)

            res = ((self.basis_train[0][:, self.idx_train[idx]]).t().mm((aug_fac))).real
            res[
                res <= 0] = 0.000001  # Note here: res can take negative values, take max{0,red} -> res[res < 0] = 0, Ensure positivity
            res = torch.Tensor(c[0, :]) * res  # Normalize conditional density
            res = torch.log(res)

            for n in range(1, self.ndims):
                aug_fac = torch.complex(torch.cat((self.factors[n].real, torch.ones(1, self.F), self.factors[n].real.__reversed__()), dim=0),
                                    torch.cat((self.factors[n].imag, torch.zeros(1, self.F), self.factors[n].imag.neg().__reversed__()), dim=0))
                tmp = (((self.basis_train[n][:, self.idx_train[idx]]).t().mm((aug_fac)))).real
                tmp[tmp <= 0] = 0.000001
                tmp = torch.log(tmp)

                tmp = torch.Tensor(c[n, :]) * tmp  # Normalize conditional density
                res = res + tmp


            # Ensure sum to 1 constraint

            res = torch.exp(res)
            fnl_results = torch.sum((res.mm(torch.diagflat(self.factors[-1]))), dim=1)

            return fnl_results

        else:
            aug_fac = torch.complex(torch.cat((self.factors[0].real, torch.ones(1, self.F), self.factors[0].real.__reversed__()), dim=0),
                                    torch.cat((self.factors[0].imag, torch.zeros(1, self.F), self.factors[0].imag.neg().__reversed__()), dim=0))


            # Ensure sum to 1 conditionals
            c = np.ones((self.ndims, self.F))
            for j in range(self.ndims):
                c[j] = self.get_conditionals_per_latent(j, x)

            res = ((self.basis_val[0][:, self.idx_val[idx]]).t().mm((aug_fac))).real
            res[
                res <= 0] = 0.000001  # Note here: res can take negative values, take max{0,red} -> res[res < 0] = 0, Ensure positivity
            res = torch.Tensor(c[0, :]) * res  # Normalize conditional density
            res = torch.log(res)

            for n in range(1, self.ndims):
                aug_fac = torch.complex(torch.cat((self.factors[n].real, torch.ones(1, self.F), self.factors[n].real.__reversed__()), dim=0),
                                    torch.cat((self.factors[n].imag, torch.zeros(1, self.F), self.factors[n].imag.neg().__reversed__()), dim=0))


                tmp = (((self.basis_val[n][:, self.idx_val[idx]]).t().mm((aug_fac)))).real
                tmp[tmp <= 0] = 0.000001
                # tmp[tmp > 10 ^ 12] = 10 ^ 9
                tmp = torch.Tensor(c[n, :]) * tmp  # Normalize conditional density
                tmp = torch.log(tmp)
                res = res + tmp

            # Ensure sum to 1 constraint
            res = torch.exp(res)
            fnl_results = torch.sum((res.mm(torch.diagflat(self.factors[-1]))), dim=1)
            return fnl_results


class htf_sgd:
    def __init__(self, coef_size, alpha, F, lr, b_size, max_iter=100, tol=1e-3):

        self.coef_size = coef_size
        self.alpha = alpha
        self.F = F
        self.l_rate = lr
        self.b_size = b_size
        self.max_iter = max_iter
        self.tol = tol
        self.log_val = float('inf')
        self.factors = []
        self.scale = 1 / (self.F) * np.ones(self.F)
        self.N = None
        self.max_iter_no_impr = 25  # Patience, early stopping after no improvements , 25 iterations
        self.md = None

    def fit(self, X_train, X_val):

        s_tr, _ = X_train.shape
        s_vl, self.N = X_val.shape

        X_train = torch.from_numpy(X_train).float()  # Converting numpy array to tensor
        X_val = torch.from_numpy(X_val).float()

        train_dataset = TensorDataset(X_train)  # [8505, 82]
        val_dataset = TensorDataset(X_val)

        self.md = csid_net(train_dataset.tensors[0], val_dataset.tensors[0], self.F, [self.coef_size] * self.N,
                           self.b_size)

        cost_hist = deque([float('Inf')], maxlen=self.max_iter_no_impr)
        train_dataloader = DataLoader(train_dataset, batch_size=self.b_size, shuffle=False, num_workers=4)
        val_dataloader = DataLoader(val_dataset, batch_size=self.b_size, shuffle=False, num_workers=4)
        optimizer = optim.Adam(self.md.parameters(), lr=self.l_rate, weight_decay=self.alpha)

        for i in range(self.max_iter):
            print('Iteration', i)
            train(self.md, optimizer, train_dataloader)
            self.log_val = validate(self.md, val_dataloader, s_vl)
            if i > self.max_iter_no_impr and self.log_val > max(cost_hist):
                break
            cost_hist.append(self.log_val)
        self.factors = [fact for fact in self.md.factors]

