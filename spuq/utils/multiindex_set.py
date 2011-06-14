class MultiindexSet(object):
    def __init__(self, arr):
        self.arr=arr
        self.m=arr.shape[1]
        self.p=max(arr.sum(1))
        pass

    @staticmethod
    def createCompleteOrderSet(m,p):
        def createMultiindexSet(m, p):
            from numpy import int8,  zeros,  vstack, hstack
            if m==0:
                return zeros( (1, 0),  int8 )
            else:
                I=zeros( (0, m),  int8 )
                for q in xrange(0, p+1):
                    J=createMultiindexSet(m-1, q)
                    Jn=q-J.sum(1).reshape((J.shape[0],1))
                    I=vstack( (I,  hstack( (J, Jn))))
                return I
        arr=createMultiindexSet(m,p)
        return MultiindexSet(arr)
        
    def __repr__(self):
        return "MI(m="+str(self.m)+",p="+str(self.p)+",p="+str(self.arr)+")"
        

def __main__( ):
    print "Multiindex: ",  MultiindexSet.createCompleteOrderSet( 3,  3)
__main__()
