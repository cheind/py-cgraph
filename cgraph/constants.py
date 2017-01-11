from cgraph.arithmetics import ArithmeticNode
import math

class Constant(ArithmeticNode):
    def __init__(self, value):
        super(Constant, self).__init__()
        self.value = value

    def __str__(self):
        return str(self.value)
    
    def forward(self, inputs):
        return self.value, None

    def ngradient(self, cache):
        pass

    def sgradient(self, inputs):
        pass

class NamedConstant(Constant):
    def __init__(self, name, value):
        super(NamedConstant, self).__init__(value)
        self.name = name

    def __str__(self):
        return str(self.name)


pi = NamedConstant('pi', math.pi)