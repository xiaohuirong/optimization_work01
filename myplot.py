#!/bin/python

import numpy as np
import matplotlib.pyplot as plt

N = 110
J = 10
I = 5


class Myfigure:
    def __init__(self, data_clu, data_nd) -> None:
        fig, self.ax = plt.subplots()
        self.ax.scatter(
            data_clu[..., 0],
            data_clu[..., 1],
            alpha=1,
            edgecolors="none",
            marker="^",
            linewidths=1.5,
        )

        angle = np.arange(0, 2 * np.pi, 0.01)

        for j in range(J):
            clu_x = data_clu[j][0]
            clu_y = data_clu[j][1]
            clu_r = data_clu[j][2]
            tmp_x = clu_x + clu_r * np.cos(angle)
            tmp_y = clu_y + clu_r * np.sin(angle)
            self.ax.plot(tmp_x, tmp_y, color="red", linestyle="dashed")

        for j in range(J):
            plt.scatter(data_nd[j, ..., 0], data_nd[j, ..., 1], alpha=0.5)

        self.ax.set(xlabel="x", ylabel="y", title="Trajectory")
        self.ax.set_xlim(0, 2000)
        self.ax.set_ylim(0, 2000)
        self.ax.set_aspect("equal", "box")

    def add(self, q):
        self.ax.plot(q[..., 0], q[..., 1], alpha=0.5)
        self.ax.scatter(q[..., 0], q[..., 1], alpha=0.3, edgecolors="none")

    def show(self):
        plt.show()


if __name__ == "__main__":
    from scipy.io import loadmat

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

    figure = Myfigure(data_clu, w)
    figure.add(q_)
    figure.show()
