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

to = TO(w, omega, p, q, t, tau)
[result, q] = to.opt()
figure.add(q)

so = SO(w, omega, p, q, t)
[result, t, tau] = so.opt()
print(t)
print(tau)
np.savetxt("tau.csv", tau, delimiter=",")

po = PO(w, omega, q, tau)
[result, p] = po.opt()

to = TO(w, omega, p, q, t, tau)
[result, q] = to.opt()
figure.add(q)


figure.show()
