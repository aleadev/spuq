class Lookup(dict):
    """
    a dictionary which can lookup value by key, or keys by value
    """
    def __init__(self, items=[]):
        """items can be a list of pair_lists or a dictionary"""
        dict.__init__(self, items)
        self._update()

    def _update(self):
        """update inverse map"""
        self._invdict = dict([(v,k) for k,v in self.iteritems()])

    def __setitem__(self, key, value):
        super(Lookup, self).__setitem__(key, value)
        update()        

    def get_key(self, value):
        """find the key given a value"""
        return self._invdict[value]

    def get_value(self, key):
        """find the value given a key"""
        return self[key]
