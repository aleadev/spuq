from numpy import *
from pylab import plot, show, axis, subplot, xlabel, ylabel, grid, hist, figure

class Plotter(object):
    def __init__(self):
        pass
    
    @staticmethod
    def scatterplot(indices,data,grid=True):
        x = unique(indices[:,0])
        y = unique(indices[:,1])
        M = len(y)
        N = len(x)
        for ai,bi in indices:
            xi = nonzero(x==ai)[0][0]
            yi = nonzero(y==bi)[0][0]
            print 'subplot: ',(ai,bi),(xi,yi),(M,N),N*(M-1)+xi+1-yi*N
            subplot(M,N,N*(M-1)+xi+1-yi*N)
            plot(data[ai,:], data[bi,:], 'b.')
            if xi == 0:
                ylabel('y['+str(bi)+']')
            if yi == 0:
                xlabel('x['+str(ai)+']')
        axis('equal')
#        grid(True)
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
