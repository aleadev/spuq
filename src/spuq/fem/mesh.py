from abc import ABCMeta, abstractmethod


class FEMMesh(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def refine(self, faces):
        pass
