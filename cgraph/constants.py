import cgraph.arithmetics
import math

class Constant(cgraph.arithmetics.ArithmeticNode):
    def __init__(self, value):
        super(Constant, self).__init__()
        self._value = value

    def __str__(self):
        return str(self._value)
    
    def value(self, ctx):
        ctx.value = self._value

    # Do we want constants to be unique?
    # Both ways work, but generate different graphs.
    
    # def __hash__(self):
    #     return hash(self._value)            
    
    # def __eq__(self, other):
    #     if isinstance(other, self.__class__):
    #         return self._value == other._value      
    #     else:
    #         return False

    def ngradient(self, ctx):
        pass

    def sgradient(self, ctx):
        pass

class NamedConstant(Constant):
    def __init__(self, name, value):
        super(NamedConstant, self).__init__(value)
        self.name = name

    def __str__(self):
        return str(self.name)


pi = NamedConstant('pi', math.pi)