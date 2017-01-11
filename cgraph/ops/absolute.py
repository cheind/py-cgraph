from ..arithmetics import ArithmeticNode
from ..constants import Constant
from ..graphs import graph

class Abs(ArithmeticNode):

    def value(self, ctx):
        ctx.value = abs(ctx.in_values[0])

    def ngradient(self, ctx):
        ctx.ngradient = ctx.in_values[0] / abs(ctx.in_values[0])

    def __str__(self):
        e = graph.in_edges(self)
        return 'abs({})'.format(e[0][0])