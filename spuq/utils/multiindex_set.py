class MultiindexSet(object):
    def __init__(self, m, p):
        pass
        
def createMultiindexSet(m, p,  exact=False):
    from numpy import int8,  zeros,  vstack, hstack
    if m==0:
        if p==0:
            return zeros( (1, 0),  int8 )
        else:
            return zeros( (0, 0),  int8 )
    else:
        I=zeros( (0, m),  int8 )
        if exact:
            for r in xrange(0, p+1):
                J=createMultiindexSet(m-1, r,  True)
                I=vstack( (I,  hstack( (J, (p-r)+zeros( (J.shape[0], 1), int8 )))))
        else:
            for q in xrange(0, p+1):
                for r in xrange(0, q+1):
                    J=createMultiindexSet(m-1, r,  True)
                    I=vstack( (I,  hstack( (J, (q-r)+zeros( (J.shape[0], 1), int8 )))))
        return I
                
        

def __main__( ):
    #print "Multiindex: ",  createMultiindexSet( 1,  0)
    #print "Multiindex: ",  createMultiindexSet( 1,  1)
    #print "Multiindex: ",  createMultiindexSet( 1,  2)
    #print "Multiindex: ",  createMultiindexSet( 1,  3)
    #return
    #print "Multiindex: ",  createMultiindexSet( 0,  3)
    #print "Multiindex: ",  createMultiindexSet( 1,  3)
    print "Multiindex: ",  createMultiindexSet( 3,  3)
    
__main__()
        
        

        
