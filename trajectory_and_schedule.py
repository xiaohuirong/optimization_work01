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


class TSO:
    def __init__(self, chi, ksi, w, omega, p, q_) -> None:
        super().__init__()
        self.chi = chi
        self.ksi = ksi
        self.w = w
        self.omega = omega
        self.p = p
        self.q_ = q_

        self.q = cp.Variable((N, 2))
        self.tau = cp.Variable((N, J))
        self.t = cp.Variable(N)
        self.E = cp.Variable()
        self.R_ = 0

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
        for j in range(J):
            for i in range(I):
                self.R_ -= cp.sum_squares(
                    cp.multiply(
                        np.expand_dims(self.sqrt_B_[..., j, i], 1),
                        np.expand_dims(self.w[j][i], 0) - self.q,
                    )
                )
        self.R_ += np.sum(self.B_ * self.horizon_d2_)
        self.R_ += cp.sum(cp.multiply(self.tau, self.A_))

    def opt(self):
        self.cal_B_()
        self.cal_A_()
        self.cal_R_()

        objective = cp.Maximize(self.R_ + self.ksi * self.chi * self.E)

        constraints = [
            cp.sum(self.tau, axis=1) <= self.t,
            self.tau >= 0,
            self.t >= 0,
            self.t <= 1,
            self.E >= 0,
            self.E <= 1,
            self.q >= 0,
            self.q <= 2000,
        ]
        prob = cp.Problem(objective, constraints)
        result = prob.solve()
        return result, self.q.value, self.t.value, self.tau.value, self.E


if __name__ == "__main__":
    q_ = np.random.rand(N, 2) * 2000
    w = np.random.rand(J, I, 2) * 2000
    omega = np.random.rand(N, J)
    p = np.random.rand(N, J, I)
    tso = TSO(1, 1, w, omega, p, q_)
    tso.opt()
