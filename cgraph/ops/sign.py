import math
from numbers import Number

from ..node import Node
from ..arithmetics import ArithmeticNode
from ..constants import Constant
from ..graphs import graph
from ..helpers import nary_link

class Signum(ArithmeticNode):
     
    def value(self, ctx):
        ctx.value = sgn(ctx.in_values[0])

    def ngradient(self, ctx):
        ctx.ngradient = 0.

    def sgradient(self, ctx):
        ctx.sgradient = Constant(0)

    def __str__(self):
        e = graph.in_edges(self)
        return 'sgn({})'.format(e[0][0])

def sgn(x):
    if isinstance(x, Number):
        return math.copysign(1, x)
    elif isinstance(x, Node):
        return nary_link(Signum(), x)
    else:
        raise TypeError('Unsupported type {}'.format(type(x)))

