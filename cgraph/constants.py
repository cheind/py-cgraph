from cgraph.arithmetics import ArithmeticNode
import math

class Constant(ArithmeticNode):
    def __init__(self, value):
        super(Constant, self).__init__()
        self._value = value

    def __str__(self):
        return str(self._value)
    
    def value(self, ctx):
        ctx.value = self._value

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