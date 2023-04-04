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


class TO:
    def __init__(self, w, omega, p, q_, t, tau) -> None:
        super().__init__()
        self.w = w
        self.omega = omega
        self.p = p
        self.q_ = q_
        self.t = t
        self.tau = tau

        self.q = cp.Variable((N, 2))
        self.R_ = 0
        self.clu_R_ = list()
        self.constraints = list()

    def cal_d_(self):
        self.d = np.zeros((N, J, I))
        for n in range(N):
            for j in range(J):
                for i in range(I):
                    self.d[n, j, i] = np.linalg.norm(
                        np.append(self.q_[n], H) - np.append(self.w[j][i], 0)
                    )

    def cal_delta(self):
        self.cal_d_()
        self.theta = 180 / np.pi * np.arcsin(H / self.d)
        self.delta = (
            20 * np.log10(4 * np.pi * f / c)
            + (etaLoS - etaNlos) / (1 + a * np.exp(-b * (self.theta - a)))
            + etaNlos
        )

    def cal_ell(self):
        self.cal_delta()
        self.ell = (1 / self.d**2) * (10 ** (self.delta / 10))

    def cal_gama(self):
        self.cal_delta()
        self.gama = (
            1
            / sigma
            * K
            * self.p
            * np.expand_dims(1 / self.omega, 2)
            * (1 / (10 ** (self.delta / 10)))
        )

    def cal_d2_(self):
        self.d2_ = np.zeros((N, J, I))
        self.horizon_d2_ = np.zeros((N, J, I))
        for n in range(N):
            for j in range(J):
                for i in range(I):
                    self.horizon_d2_ = np.sum((self.w[j][i] - self.q_[n]) ** 2)

        self.d2_ = self.horizon_d2_ + H**2

    def cal_B_(self):
        self.cal_gama()
        self.cal_d2_()
        up = self.gama * np.log2(np.e)
        down = (self.gama + self.d2_) * (self.gama)
        self.B_ = up / down

    def cal_A_(self):
        tmp = np.log2(1 + self.gama / self.d2_)
        part_a = np.sum(tmp, axis=2)
        part_b = K * np.log2(self.omega) - K * np.log2(np.e) * (1 - 1 / self.omega)
        self.A_ = part_a + part_b

    def cal_R_(self):
        self.sqrt_B_ = np.sqrt(self.B_)
        self.clu_R_ = np.sum(self.B_ * self.horizon_d2_, (0, 2))
        self.clu_R_ = list(self.clu_R_)
        for j in range(J):
            for i in range(I):
                R_ = cp.sum_squares(
                    cp.multiply(
                        np.expand_dims(self.sqrt_B_[..., j, i], 1),
                        np.expand_dims(self.w[j][i], 0) - self.q,
                    )
                )
                self.clu_R_[j] -= R_
            self.R_ += self.clu_R_[j]

    def cal_constraints(self):
        start_and_end = [
            self.q[0] == self.q_[0],
            self.q[N - 1] == self.q_[N - 1],
        ]
        d_max = [
            cp.sum(cp.square(self.q[1:N] - self.q[0 : N - 1]), 1) <= Dmax**2,
        ]
        speed_max = [
            cp.sum(cp.square(self.q[1:N] - self.q[0 : N - 1]), 1)
            <= self.t[..., 0] * Vmax**2,
        ]
        border = [
            self.q >= 0,
            self.q <= 2000,
        ]

        close_cons = [cp.sum(cp.square(self.q - self.q_), 1) <= 100**2]

        rate_min = [self.clu_R_[j] >= 420 for j in range(J)]

        test_cons = [self.q == np.expand_dims([655, 308], 0)]

        self.constraints.extend(start_and_end)
        self.constraints.extend(speed_max)
        self.constraints.extend(d_max)
        self.constraints.extend(border)
        # self.constraints.extend(close_cons)
        # self.constraints.extend(rate_min)
        # self.constraints.extend(test_cons)

    def opt(self):
        self.cal_B_()
        self.cal_A_()
        self.cal_R_()

        objective = cp.Maximize(self.R_)
        # clu = 10
        # objective = cp.Maximize(self.clu_R_[clu - 1])

        self.cal_constraints()

        prob = cp.Problem(objective, self.constraints)
        result = prob.solve()
        return result, self.q.value


if __name__ == "__main__":
    from scipy.io import loadmat
    import matplotlib.pyplot as plt

    init_data = loadmat("initData.mat")
    qx = init_data["qx"]
    qy = init_data["qy"]
    data_clu = init_data["dataClu"]
    data_nd = init_data["dataNd"]
    t = init_data["t"]
    tau = init_data["tau"]
    tau = np.vstack([tau, np.zeros(J)])

    w = np.zeros((J, I, 2))
    for j in range(J):
        for i in range(I):
            w[j][i] = data_nd[i, ..., j]

    q_ = np.column_stack((qx, qy))

    omega = np.random.rand(N, J)
    p = np.random.rand(N, J, I)

    tso = TO(w, omega, p, q_, t, tau)
    [result, q_new] = tso.opt()

    print(result)
    for j in range(J):
        print(j + 1, "= ", tso.clu_R_[j].value)

    fig, ax = plt.subplots()
    ax.scatter(q_[..., 0], q_[..., 1])
    ax.scatter(q_new[..., 0], q_new[..., 1])
    plt.show()
