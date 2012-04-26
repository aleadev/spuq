class ParametricArray(object):
    Empty = object()

    def __init__(self, func):
        self._vals = []
        self._func = func


    def __len__(self):
        return len(self._vals)

    def __getitem__(self, i):
        if i >= len(self._vals):
            self._grow(i)
        val = self._vals[i]
        if val is ParametricArray.Empty:
            val = self._func(i)
            self._vals[i] = val
        return val

    def _grow(self, i):
        l = len(self._vals)
        if i >= l:
            self._vals += [ParametricArray.Empty] * (i - l + 1)

