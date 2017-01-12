from ..arithmetics import ArithmeticNode
from ..constants import Constant
from ..graphs import graph
from ..helpers import nary_link

class Mul(ArithmeticNode):

    def value(self, ctx):
        ctx.value = ctx.in_values[0] * ctx.in_values[1]        

    def ngradient(self, ctx):
        ctx.ngradient = [ctx.in_values[1], ctx.in_values[0]]

    def sgradient(self, ctx):
        ctx.sgradient = [ctx.in_nodes[1], ctx.in_nodes[0]]

    def __str__(self):
        e = graph.in_edges(self)
        return '({}*{})'.format(e[0][0], e[1][0])


def sym_mul(x, y):
    return nary_link(Mul(), x, y)