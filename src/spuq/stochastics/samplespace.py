from __future__ import division


class SampleSpace:
    def __init__(self, rvs):
        self.rvs = rvs
    
    def __getitem__(self, i):
        return self.rvs(i)
    