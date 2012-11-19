import time
from spuq.utils.contextdecorator import ContextDecorator
from collections import defaultdict

def timing(msg="", logfunc=None, strformat=None):
    class TimingCtx(object):
        @staticmethod 
        def printer(msg):
            print msg

        def __init__(self, msg="", logfunc=None, strformat=None):
            self.msg = msg

            if logfunc is None:
                self.logfunc = printer
            else:
                self.logfunc = logfunc

            if strformat is None:
                self.strformat = "Elapsed time: %s sec (%s)"
            else:
                self.strformat = strformat

        def __enter__(self):
            self.start = time.clock()

        def __exit__(self, *args):
            elapsed = time.clock() - self.start
            msg = self.strformat % (elapsed, self.msg)
            self.logfunc(msg)

    return TimingCtx(msg, logfunc, strformat)


_TIMINGS_ = defaultdict(list)
_no_timing_ = False

class timing2(ContextDecorator):
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        global _no_timing_
        if not _no_timing_:
            self.start = time.clock()

    def __exit__(self, *args):
        global _TIMINGS_
        global _no_timing_
        if not _no_timing_:
            dt = time.clock() - self.start
            _TIMINGS_[self.name].append(dt)

    @staticmethod
    def get_timings():
        from operator import itemgetter
        global _TIMINGS_
        T = ((k,sum(v)) for k,v in _TIMINGS_.iteritems())
        return sorted(T, key=itemgetter(1), reverse=True)
        
    @staticmethod
    def get_timings_str():
        T = timing2.get_timings()
        s = "\n" + "*"*40 + "\nTIMINGS\n"
        s += "\n".join(str(k)+"\t:\t"+str(v) for k,v in T)
        s += "\n" + "*"*40
        return s


#from time import sleep
#class TestA(object):
#    @timing2("f1")
#    def f1(self):
#        i = 1
#        for _ in range(100):
#            i *= i
#        return i

#    @timing2("f2")
#    def f2(self):
#        i = 1
#        for _ in range(1000):
#            i *= i
#        return i

#A = TestA()
#for _ in range(1000):
#    A.f1()
#    A.f2()
#    A.f1()
#print timing2.get_timings_str()
