import math

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
    return math.copysign(1, x)

def sym_sgn(x):
    return nary_link(Signum(), x)

