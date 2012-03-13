from abc import ABCMeta, abstractmethod

class MathObject:
    __metaclass__ = ABCMeta
    
    @abstractmethod
    def copy(self):
        raise NotImplementedError

    def __ne__(self, other):
        return not self.__eq__(other)
    
    
