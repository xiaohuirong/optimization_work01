#!/bin/python
import numpy as np

N = 110
J = 10
I = 5

H = 150
c = 3e8
f = 2e9
etaLoS = 0.1
etaNlos = 21
a = 5.0188
b = 0.3511
sigma = 1e-10
K = 5
Vmax = 30
Dmax = 100

v_0 = 4.03
P_0 = 79.86
U_tip = 120
P_i = 88.63
d_0 = 0.6
rho = 1.225
s = 0.05
A = 0.503

p_0 = 0.02
pMsk = 0.6
pMax = 2
Dmin = 500


class RO:
    def __init__(self, w, q, tau, p) -> None:
        super().__init__()
        self.ksi = 1
        self.q = q
        self.tau = tau
        self.p = p
        self.w = w
        self.omega = np.full((N, J), 1.0)
        self.tol = 10e-5
        self.maxi = 1000
        self.cal_ell()

    def cal_d(self):
        self.d = np.zeros((N, J, I))
        for n in range(N):
            for j in range(J):
                for i in range(I):
                    self.d[n, j, i] = np.linalg.norm(
                        np.append(self.q[n], H) - np.append(self.w[j][i], 0)
                    )

    def cal_delta(self):
        self.cal_d()
        self.theta = 180 / np.pi * np.arcsin(H / self.d)
        self.delta = (
            20 * np.log10(4 * np.pi * f / c)
            + (etaLoS - etaNlos) / (1 + a * np.exp(-b * (self.theta - a)))
            + etaNlos
        )

    def cal_ell(self):
        self.cal_delta()
        self.ell = (1 / self.d**2) * (10 ** (self.delta / 10))
        self.up = self.p * self.ell

    def g(self, n, j):
        up = self.up[n, j]
        down = sigma + K * up / self.omega[n, j]
        result = 1 + np.sum(up / down)
        return result

    def opt(self, n, j):
        print(self.up[n, j])
        for iter in range(self.maxi):
            new_value = self.g(n, j)
            error = abs(new_value - self.omega[n, j])
            rerror = error / (new_value + 10e-5)
            if error <= self.tol and rerror <= self.tol:
                self.omega[n, j] = new_value
                break
            self.omega[n, j] = new_value
        print(self.omega[n, j])

    def opt_all(self):
        for n in range(N):
            for j in range(J):
                self.opt(n, j)


if __name__ == "__main__":
    from scipy.io import loadmat
    import matplotlib.pyplot as plt

    init_data = loadmat("initData.mat")
    qx = init_data["qx"]
    qy = init_data["qy"]
    data_clu = init_data["dataClu"]
    data_nd = init_data["dataNd"]
    t = init_data["t"]
    t = t[:, 0]
    tau = init_data["tau"]
    tau = np.vstack([tau, np.zeros(J)])

    w = np.zeros((J, I, 2))
    for j in range(J):
        for i in range(I):
            w[j][i] = data_nd[i, :, j]

    q_ = np.column_stack((qx, qy))

    p = np.random.rand(N, J, I)

    tso = RO(w, q_, tau, p)
    tso.opt_all()
