import math
from numbers import Number

from ..node import Node
from ..arithmetics import ArithmeticNode
from ..constants import Constant
from ..graphs import graph
from ..helpers import nary_link

import cgraph.guards as guards

class Logartihm(ArithmeticNode):

    def value(self, ctx):
        ctx.value = math.log(guards.zn(ctx.in_values[0]))

    def ngradient(self, ctx):
        ctx.ngradient = 1. / guards.z(ctx.in_values[0])

    def sgradient(self, ctx):
        ctx.sgradient = Constant(1) / ctx.in_nodes[0]
    
    def __str__(self):
        e = graph.in_edges(self)
        return 'log({})'.format(e[0][0])

def sym_log(x):
    return nary_link(Logartihm(), x)