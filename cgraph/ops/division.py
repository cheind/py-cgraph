from ..arithmetics import ArithmeticNode
from ..constants import Constant
from ..graphs import graph
from ..helpers import nary_link

import cgraph.guards as guards

class Div(ArithmeticNode):

    def value(self, ctx):
        ctx.value = ctx.in_values[0] / guards.z(ctx.in_values[1])

    def ngradient(self, ctx):
        ctx.ngradient = [
            1. / guards.z(ctx.in_values[1]), 
            -ctx.in_values[0]/guards.z(ctx.in_values[1])**2
        ]

    def sgradient(self, ctx):
        ctx.sgradient = [
            Constant(1) / ctx.in_nodes[1], 
            -ctx.in_nodes[0]/ctx.in_nodes[1]**2
        ]

    def __str__(self):
        e = graph.in_edges(self)
        return '({}/{})'.format(e[0][0], e[1][0])

def sym_div(x, y):
    return nary_link(Div(), x, y)