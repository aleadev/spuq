from scipy.special import zeta
import numpy as np
#from spuq.application.egsz.sample_problems import SampleProblem
#import matplotlib.pyplot as plt
import pylab as pl


np.set_printoptions(suppress=True, linewidth=1000, precision=3, edgeitems=20)


class SampleProblem(object):
    @staticmethod
    def get_decay_start(exp, gamma=1):
        start = 1
        while zeta(exp, start) >= gamma:
            start += 1
        return start


def compute_decay(N, decayexp, mode, gamma, with1=True):
    if mode == 2:
        start = SampleProblem.get_decay_start(decayexp, gamma)
        A = gamma / zeta(decayexp, start)
    elif mode == 1:
        start = 1
        A = gamma / zeta(decayexp, start)
    assert mode == start

    if with1:
        a = np.zeros(N+1)
        a[0] = 1
        for i in range(N):
            m = i + start
            a[i+1] = A * (m**(-decayexp))
    else:
        a = np.zeros(N)
        for i in range(N):
            m = i + start
            a[i] = A * (m**(-decayexp))

    return a

gamma = 0.9

color = {2:"g", 4:"r"}
marker = {1:"o", 2:"x"}

N = 15
l = []
pl.subplot(2,2,1)
for sigma in [2, 4]:
    for mode in [1, 2]:
        dc = compute_decay(N, sigma, mode, gamma, False)
        dc100 = compute_decay(100, sigma, mode, gamma, False)
        print sigma, mode, dc, sum(dc100)
        pl.plot(dc,color[sigma]+marker[mode]+"-")
        l.append("sigma=%d, start=%d" % (sigma, mode))
pl.legend(l)

N = 6
l = []
pl.subplot(2,2,2)
for sigma in [2, 4]:
    for mode in [1, 2]:
        dc = compute_decay(N, sigma, mode, gamma, False)
        pl.plot(dc,color[sigma]+marker[mode]+"-")
        l.append("sigma=%d, start=%d" % (sigma, mode))
pl.legend(l)


N = 15
l = []
pl.subplot(2,2,3)
for sigma in [2, 4]:
    for mode in [1, 2]:
        dc = compute_decay(N, sigma, mode, gamma, False)
        pl.plot(pl.cumsum(dc),color[sigma]+marker[mode]+"-")
        l.append("sigma=%d, start=%d" % (sigma, mode))
pl.legend(l, loc="lower right")

N = 50
l = []
pl.subplot(2,2,4)
for sigma in [2, 4]:
    for mode in [1, 2]:
        dc = compute_decay(N, sigma, mode, gamma, False)
        pl.plot(pl.cumsum(dc),color[sigma]+marker[mode]+"-")
        l.append("sigma=%d, start=%d" % (sigma, mode))
pl.legend(l, loc="lower right")



pl.show()



