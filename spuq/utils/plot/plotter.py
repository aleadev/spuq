from numpy import *
from pylab import plot, show, axis, subplot, xlabel, ylabel, grid, hist, figure

class Plotter(object):
    def __init__(self):
        pass
    
    @staticmethod
    def scatterplot(indices,data,grid=True):
        m = unique(indices[:,0])
        n = unique(indices[:,1])
        M = len(m)
        N = len(n)
        for ai,bi in indices:
            mi = nonzero(m==ai)[0][0]
            ni = nonzero(n==bi)[0][0]
            print 'subplot: ',(ai,bi),(mi,ni),(M,N),mi*N+ni
            subplot(M,N,mi*N+ni+1)
            plot(data[indices[ai],:], data[indices[bi],:], 'b.')
            if mi == M-1:
                xlabel('x['+str(ai)+']')
            if ni == 0:
                ylabel('y['+str(bi)+']')
            axis('equal')
#            grid(True)
        show()

    @staticmethod
    def histplot(data, bins=10, normed=False, weights=None, cumulative=False, bottom=None, histtype='bar'):
        M = data.shape[0]
        N = 1
        for i in range(M-1):
            subplot(M,N,i+1)
            hist(data[i,:], bins=bins, normed=normed, weights=weights, cumulative=cumulative, bottom=bottom, histtype=histtype)

    @staticmethod
    def figure(num=1):
        figure(num)
