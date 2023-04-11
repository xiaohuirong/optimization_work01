#!/bin/python

import numpy as np
import cvxpy as cp

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


class PO:
    def __init__(self, w, omega, q, tau) -> None:
        super().__init__()
        self.w = w
        self.omega = omega
        self.ksi = 1
        self.q = q
        self.tau = tau

        self.p = cp.Variable((N, I), nonneg=True)
        self.A = cp.Parameter((N, I), nonneg=True)
        self.tauj = cp.Parameter(N, nonneg=True)

        self.cal_target()

        self.constraints = list()
        self.gen_constraints()

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

    def cal_parameter(self, j):
        self.cal_ell()
        self.A.value = (
            1
            / sigma
            * K
            * self.ell[:, j, :]
            * np.expand_dims((self.omega[:, j]) ** (-1), 1)
        )
        self.tauj.value = self.tau[:, j]

    def cal_target(self):
        R_n = cp.sum(cp.log(1 + cp.multiply(self.A, self.p)), 1)
        self.R = self.tauj @ R_n

        E_n = cp.sum(self.p + p_0, 1)
        self.E = self.tauj @ E_n
        self.target = np.log2(np.e) * self.R - self.ksi * self.E

    def gen_constraints(self):
        max_energy = [
            self.p <= pMsk,
            cp.sum(self.p, 1) <= pMax,
        ]
        min_rate = [
            self.R >= Dmin,
        ]
        self.constraints.extend(max_energy)
        self.constraints.extend(min_rate)

    def opt(self):
        objective = cp.Maximize(self.target)

        prob = cp.Problem(objective, self.constraints)

        results = []
        p_new = np.zeros((N, J, I))
        for j in range(J):
            self.cal_parameter(j)
            result = prob.solve(solver="MOSEK", verbose=False)
            results.append(result)
            p_new[:, j, :] = self.p.value

        return results, p_new


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

    # omega = np.random.rand(N, J) + 1
    omega = np.full((N, J), 1.0)
    p = np.random.rand(N, J, I)

    tso = PO(w, omega, q_, tau)
    [result, p_new] = tso.opt()
    print(result)
    print(p_new)
