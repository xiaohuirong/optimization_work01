#!/bin/python

import numpy as np
from scipy.io import loadmat

N = 110
J = 10
I = 5


def loaddata():
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

    omega = np.full((N, J), 1.0)
    p = np.random.rand(N, J, I)
    return data_clu, w, q_, omega, p, t, tau
