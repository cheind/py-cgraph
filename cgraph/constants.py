from cgraph.arithmetics import ArithmeticNode
import math

class Constant(ArithmeticNode):
    def __init__(self, value):
        super(Constant, self).__init__()
        self.value = value

    def __str__(self):
        return str(self.value)

    def __hash__(self):
        return hash(self.value)            
    
    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.value == other.value      
        else:
            return False
    
    def forward(self, inputs):
        return self.value, None

    def ngradient(self, cache):
        return [0.]

class NamedConstant(Constant):
    def __init__(self, name, value):
        super(NamedConstant, self).__init__(value)
        self.name = name

    def __str__(self):
        return str(self.name)


pi = NamedConstant('pi', math.pi)