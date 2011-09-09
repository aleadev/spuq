class ShiftedDistribution(ProbabilityDistribution):
    def __init__(self,dist,delta):
        self.dist=dist
        self.delta=delta
    def mean(self):
        return dist.mean()+delta
    def var(self):
        return dist.var()
    def __repr__(self):
        return self.dist.__repr__()+"+"+str(self.delta)
    
