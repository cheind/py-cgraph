
from numbers import Number
from cgraph.node import Node
from cgraph.graphs import graph

def wrap_args(func):
    """Decorator to convert wrap arguments into Nodes when required."""
    def wrapper(*args):
        newargs = []
        for n in args:               
            if isinstance(n, Number):
                from cgraph.constants import Constant
                newargs.append(Constant(n))
            else:
                newargs.append(n)
        
        return func(*newargs)
    return wrapper

class ArithmeticNode(Node):

    @wrap_args
    def __add__(self, other):
        return Node.nary_function(Add, self, other)

    @wrap_args
    def __sub__(self, other):
        return Node.nary_function(Sub, self, other)
    
    @wrap_args
    def __mul__(self, other):
        return Node.nary_function(Mul, self, other)
    
    @wrap_args
    def __truediv__(self, other):
        return Node.nary_function(Div, self, other)

class Add(ArithmeticNode):
    
    def compute(self, inputs):
        return inputs[0] + inputs[1], [1., 1.]

    def __str__(self):
        e = graph.in_edges(self)
        return '({} + {})'.format(e[0][0], e[1][0])

class Sub(ArithmeticNode):
    
    def compute(self, inputs):
        return inputs[0] - inputs[1], [1., -1.]

    def __str__(self):
        e = graph.in_edges(self)
        return '({} - {})'.format(e[0][0], e[1][0])

class Mul(ArithmeticNode):

    def compute(self, inputs):
        return inputs[0] * inputs[1], [inputs[1], inputs[0]]

    def __str__(self):
        e = graph.in_edges(self)
        return '({}*{})'.format(e[0][0], e[1][0])

class Div(ArithmeticNode):

    def compute(self, inputs):
        return inputs[0] / inputs[1], [1. / inputs[1], -inputs[0]/inputs[1]**2]

    def __str__(self):
        e = graph.in_edges(self)
        return '({}/{})'.format(e[0][0], e[1][0])
