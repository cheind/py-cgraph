import math
from numbers import Number

from ..node import Node
from ..arithmetics import ArithmeticNode
from ..constants import Constant
from ..graphs import graph
from ..helpers import nary_link

from .logarithm import log

class Pow(ArithmeticNode):

    def value(self, ctx):
        ctx.value = ctx.in_values[0]**ctx.in_values[1]        

    def ngradient(self, ctx):
        ctx.ngradient = [
            ctx.in_values[1] * ctx.in_values[0]**(ctx.in_values[1]-1),
            ctx.value * log(ctx.in_values[0])
        ]

    def sgradient(self, ctx):
        ctx.sgradient = [
            ctx.in_nodes[1] * ctx.in_nodes[0]**(ctx.in_nodes[1]-1),
            ctx.in_nodes[0]**ctx.in_nodes[1] * log(ctx.in_nodes[0])
        ]
    
    def __str__(self):
        e = graph.in_edges(self)
        return '({})**{}'.format(e[0][0], e[1][0])


class Exp(ArithmeticNode):

    def value(self, ctx):
        ctx.value = math.exp(ctx.in_values[0])

    def ngradient(self, ctx):
        ctx.ngradient = ctx.value

    def sgradient(self, ctx):
        ctx.sgradient = self
    
    def __str__(self):
        e = graph.in_edges(self)
        return 'e**{}'.format(e[0][0])


def exp(x):
    if isinstance(x, Number):
        return math.exp(x)
    elif isinstance(x, Node):
        return nary_link(Exp(), x)
    else:
        raise TypeError('Unsupported type {}'.format(type(x)))