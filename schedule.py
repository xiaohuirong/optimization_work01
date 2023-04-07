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

Enlarge = 100


class SO:
    def __init__(self, w, omega, p, q_) -> None:
        super().__init__()
        self.w = w
        self.omega = omega
        self.p = p
        self.ksi = 1
        self.chi = 0.01
        self.q_ = q_

        self.E = cp.Variable()
        self.y = cp.Variable(N - 1)
        self.tau = cp.Variable((N, j))
        self.t = cp.Variable(N - 1)
        self.constraints = list()

        self.cal_A_()
        self.cal_y_()
        self.gen_constraints()

    def cal_d_(self):
        self.d = np.zeros((N, J, I))
        for n in range(N):
            for j in range(J):
                for i in range(I):
                    self.d[n, j, i] = np.linalg.norm(
                        np.append(self.q_[n], H) - np.append(self.w[j][i], 0)
                    )

    def cal_d2_(self):
        self.d2_ = np.zeros((N, J, I))
        self.horizon_d2_ = np.zeros((N, J, I))
        for n in range(N):
            for j in range(J):
                for i in range(I):
                    self.horizon_d2_[n, j, i] = np.sum((self.w[j][i] - self.q_[n]) ** 2)

        self.d2_ = self.horizon_d2_ + H**2

    def cal_delta(self):
        self.cal_d_()
        self.theta = 180 / np.pi * np.arcsin(H / self.d)
        self.delta = (
            20 * np.log10(4 * np.pi * f / c)
            + (etaLoS - etaNlos) / (1 + a * np.exp(-b * (self.theta - a)))
            + etaNlos
        )

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

    def cal_y_(self):
        self.Delta_ = self.q_[1:N] - self.q_[0 : N - 1]
        self.Delta2_ = np.sum(np.square(self.Delta_), 1)
        self.Delta3_ = np.power(np.linalg.norm(self.Delta_, axis=1), 3)
        self.y2_ = np.sqrt(
            self.t**4 + self.Delta2_**2 / (4 * v_0**4)
        ) - self.Delta2_ / (2 * v_0**2)
        self.y_ = np.sqrt(self.y2_)

    def cal_A_(self):
        self.cal_gama()
        self.cal_d2_()
        tmp = np.log2(1 + self.gama / self.d2_)
        part_a = np.sum(tmp, axis=2)
        part_b = K * np.log2(self.omega) - K * np.log2(np.e) * (1 - 1 / self.omega)
        self.A_ = part_a + part_b

    def cal_R_(self):
        self.R_ = cp.sum(self.tau * self.A_)
        self.clu_R_ = cp.sum(self.tau * self.A_, 0)

    def gen_constraints(self):
        frac_t = 1 / self.t
        frac_t2 = 1 / (self.t**2)
        speed_max = [
            self.Delta2_ <= self.t * Vmax**2,
        ]

        energy = [
            P_0
            * (
                self.t[n]
                + 3 / (U_tip**2) * self.Delta2_[n] * cp.quad_over_lin(1, self.t[n])
            )
            + P_i * self.y[n]
            + 0.5
            * d_0
            * rho
            * s
            * A
            * self.Delta3_[n]
            * cp.square(cp.quad_over_lin(1, self.t[n]))
            for n in range(N)
        ]

        energy_sum = cp.sum(energy)
        energy_con = [energy_sum <= self.E]

        y_con = list()
        y_con = [
            self.y2_[n] + 2 * self.y_[n] + self.Delta2_[n] / (v_0**2)
            >= cp.square(cp.quad_over_lin(self.t[n], self.y[n]))
            for n in range(N - 1)
        ]
        y_con.append(self.y >= 0)

        close_cons = [cp.sum(cp.square(self.q - self.q_), 1) <= 100**2]

        rate_min = [self.clu_R_[j] >= 350 for j in range(J)]

        self.constraints.extend(speed_max)
        self.constraints.extend(y_con)
        # self.constraints.extend(rate_min)

    def opt(self):
        objective = cp.Maximize(Enlarge**2 * (self.R_ - self.ksi * self.chi * self.E))

        prob = cp.Problem(objective, self.constraints)
        result = prob.solve(solver="ECOS", verbose=False)
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
    t = t[..., 0]
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
    print(tso.E.value)
    print(np.sum(tso.A_ * tau))

    fig, ax = plt.subplots()
    ax.scatter(q_[..., 0], q_[..., 1])
    ax.scatter(q_new[..., 0], q_new[..., 1])

    plt.show()
