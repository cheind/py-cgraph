
from cgraph.node import Node
from cgraph.helpers import nary_function

class ArithmeticNode(Node):

    def __add__(self, other):
        from cgraph.ops import Add
        return nary_function(Add, self, other)

    def __sub__(self, other):
        from cgraph.ops import Sub
        return nary_function(Sub, self, other)
    
    def __mul__(self, other):
        from cgraph.ops import Mul
        return nary_function(Mul, self, other)
    
    def __truediv__(self, other):
        from cgraph.ops import Div
        return nary_function(Div, self, other)

    def __neg__(self):
        from cgraph.ops import Neg
        return nary_function(Neg, self)

    def __abs__(self):
        from cgraph.ops import Abs
        return nary_function(Abs, self)