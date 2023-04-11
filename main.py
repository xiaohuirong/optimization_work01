#!/bin/python

import numpy as np
from trajectory import TO
from schedule import SO
from power import PO
from myplot import Myfigure
from dataload import loaddata

N = 110
J = 10
I = 5

[clu, w, q, omega, p, t, tau] = loaddata()
figure = Myfigure(clu, w)
figure.add(q)

error_num = 0

for k in range(3):
    to = TO(w, omega, p, q, t, tau)
    [result, q] = to.opt()
    figure.add(q)

    # to = TO(w, omega, p, q, t, tau)
    # [result, q] = to.opt()
    # figure.add(q)
    # np.savetxt("B.csv", to.B_[0], delimiter=",")
    # np.savetxt("d.csv", to.d_[0], delimiter=",")
    # np.savetxt("A.csv", to.A_, delimiter=",")

    # for k in range(10):
    #     to = TO(w, omega, p, q, t, tau)
    #     [result, q] = to.opt()
    #     figure.add(q)

    so = SO(w, omega, p, q, t)
    [result, t, tau] = so.opt()
    # np.savetxt("tau.csv", tau, delimiter=",")

    try:
        po = PO(w, omega, q, tau)
        [result, p] = po.opt()
        np.savetxt("p.csv", p[0], delimiter=",")
    except:
        print("Power optimization error.")
        error_num += 1

    # to = TO(w, omega, p, q, t, tau)
    # [result, q] = to.opt()
    # figure.add(q)

print("error_num", error_num)
figure.show()
