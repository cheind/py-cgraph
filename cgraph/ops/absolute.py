import math

from ..arithmetics import ArithmeticNode
from ..constants import Constant
from ..graphs import graph
from ..helpers import nary_link

from .sign import sgn

class Abs(ArithmeticNode):

    def value(self, ctx):
        ctx.value = abs(ctx.in_values[0])

    def ngradient(self, ctx):
        ctx.ngradient = sgn(ctx.in_values[0])

    def sgradient(self, ctx):
        ctx.sgradient = sgn(ctx.in_nodes[0])

    def __str__(self):
        e = graph.in_edges(self)
        return 'abs({})'.format(e[0][0])