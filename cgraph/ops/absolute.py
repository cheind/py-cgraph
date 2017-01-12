import math

from ..arithmetics import ArithmeticNode
from ..graphs import graph
from ..helpers import nary_link

from .sign import sgn, sym_sgn

class Abs(ArithmeticNode):

    def value(self, ctx):
        ctx.value = abs(ctx.in_values[0])

    def ngradient(self, ctx):
        ctx.ngradient = sgn(ctx.in_values[0])

    def sgradient(self, ctx):
        ctx.sgradient = sym_sgn(ctx.in_nodes[0])

    def __str__(self):
        e = graph.in_edges(self)
        return 'abs({})'.format(e[0][0])


def sym_abs(x):
    return nary_link(Abs(), x)