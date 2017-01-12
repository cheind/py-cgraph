import math
from numbers import Number

from ..node import Node
from ..arithmetics import ArithmeticNode
from ..graphs import graph
from ..helpers import nary_link

from .logarithm import sym_log

class Pow(ArithmeticNode):

    def value(self, ctx):
        ctx.value = ctx.in_values[0]**ctx.in_values[1]        

    def ngradient(self, ctx):
        ctx.ngradient = [
            ctx.in_values[1] * ctx.in_values[0]**(ctx.in_values[1]-1),
            ctx.value * math.log(ctx.in_values[0])
        ]

    def sgradient(self, ctx):
        ctx.sgradient = [
            ctx.in_nodes[1] * ctx.in_nodes[0]**(ctx.in_nodes[1]-1),
            ctx.in_nodes[0]**ctx.in_nodes[1] * sym_log(ctx.in_nodes[0])
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

def sym_pow(x, y):
    return nary_link(Pow(), x, y)

def sym_exp(x):
    return nary_link(Exp(), x)
