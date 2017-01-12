
from ..arithmetics import ArithmeticNode
from ..constants import Constant
from ..graphs import graph
from ..helpers import nary_link


class Add(ArithmeticNode):
    def value(self, ctx):
        ctx.value = ctx.in_values[0] + ctx.in_values[1]        

    def ngradient(self, ctx):
        ctx.ngradient = [1, 1]

    def sgradient(self, ctx):
        ctx.sgradient = [Constant(1), Constant(1)]

    def __str__(self):
        e = graph.in_edges(self)
        return '({} + {})'.format(e[0][0], e[1][0])

def sym_add(x, y):
    return nary_link(Add(), x, y)

def sym_sum(elements):
    assert len(elements) > 0    
    
    n = elements[0]
    for idx in range(1, len(elements)):
        n = n + elements[idx]
    return n