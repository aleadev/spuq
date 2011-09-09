class ForgetfulVector(object):
    def __init__(self, remember, start=0):
        self.rem=remember
        self.curr=start
        self.items=self.rem*[None]

    def __getitem__(self,k):
        if k>self.curr or k<=self.curr-self.rem:
            raise FoobarException()
        return self.items[self.curr-k]

    def __setitem__(self,k,item):
        if k>self.curr+1 or k<=self.curr-self.rem:
            raise FoobarException()
        elif k==self.curr+1:
            self.items = [item]+self.items[:-1]
            self.curr = self.curr+1
        else:
            self.items[self.curr-k] = item

    def __repr__(self):
        s=[]
        for d in xrange(self.rem):
            s=s+[str(self.curr-d)+": "+str(self.items[d])];
        return ", ".join(s)

if __name__=="__main__":
    f=ForgetfulVector(2)
    f[0]=1
    f[-1]=0
    print 0, f[0]
    for i in range(1,100): 
        f[i]=f[i-1]+f[i-2]
        print i, f[i]
        
    print f

