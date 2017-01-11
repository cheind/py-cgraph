
from cgraph.graphs import graph


class Node:       
    def __str__(self):
        return self.name if self.name is not None else self.__class__.__name__    
        
    def __repr__(self):
        return self.__str__()

    def value(self, ctx):
        raise NotImplementedError()

    def ngradient(self, ctx):
        raise NotImplementedError()

    def sgradient(self, ctx):
        raise NotImplementedError()

    def eval(self, **kwargs):
        from cgraph.eval import eval
        return eval(self, **kwargs)

    def ndiff(self, **kwargs):
        from cgraph.ndiff import ndiff
        return ndiff(self, **kwargs)

    def sdiff(self):
        from cgraph.sdiff import sdiff
        return sdiff(self)