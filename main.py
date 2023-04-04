#!/bin/python

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from trajectory_and_schedule import TSO
from myplot import show_figure

N = 110
J = 10
I = 5

init_data = loadmat("initData.mat")

qx = init_data["qx"]
qy = init_data["qy"]
data_clu = init_data["dataClu"]
data_nd = init_data["dataNd"]
t = init_data["t"]
tau = init_data["tau"]

w = np.zeros((J, I, 2))
for j in range(J):
    for i in range(I):
        w[j][i] = data_nd[i, ..., j]

q_ = np.column_stack((qx, qy))

# show_figure(q_, data_clu, w)


omega = np.random.rand(N, J)
p = np.random.rand(N, J, I)
# tso = TSO(1, 1, w, omega, p, q_)
# [result, q_new, t_new, tau_new, E_new] = tso.opt()
# print(q_new)

# show_figure(q_new, data_clu, w)
