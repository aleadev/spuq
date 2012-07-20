import time


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
