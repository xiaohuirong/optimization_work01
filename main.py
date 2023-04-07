#!/bin/python

import numpy as np
from scipy.io import loadmat
from trajectory import TO
from myplot import Myfigure
from dataload import loaddata

N = 110
J = 10
I = 5

[clu, w, q_, omega, p, t, tau] = loaddata()
figure = Myfigure(clu, w)
figure.add(q_)

tso = TO(w, omega, p, q_, t, tau)
[result, q_new] = tso.opt()
figure.add(q_new)


figure.show()
