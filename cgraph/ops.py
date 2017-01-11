from cgraph.arithmetics import ArithmeticNode
from cgraph.constants import Constant
from cgraph.graphs import graph

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

class Sub(ArithmeticNode):

    def value(self, ctx):
        ctx.value = ctx.in_values[0] - ctx.in_values[1]        

    def ngradient(self, ctx):
        ctx.ngradient = [1, -1]

    def sgradient(self, ctx):
        ctx.sgradient = [Constant(1), Constant(-1)]
    
    def __str__(self):
        e = graph.in_edges(self)
        return '({} - {})'.format(e[0][0], e[1][0])

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

class Div(ArithmeticNode):

    def value(self, ctx):
        ctx.value = ctx.in_values[0] / ctx.in_values[1]        

    def ngradient(self, ctx):
        ctx.ngradient = [1. / ctx.in_values[1], -ctx.in_values[0]/ctx.in_values[1]**2]

    def __str__(self):
        e = graph.in_edges(self)
        return '({}/{})'.format(e[0][0], e[1][0])

class Neg(ArithmeticNode):

    def value(self, ctx):
        ctx.value = -ctx.in_values[0]

    def ngradient(self, ctx):
        ctx.ngradient = -1

    def sgradient(self, ctx):
        ctx.sgradient = Constant(-1)

    def __str__(self):
        e = graph.in_edges(self)
        return '-{}'.format(e[0][0])


class Abs(ArithmeticNode):

    def value(self, ctx):
        ctx.value = abs(ctx.in_values[0])

    def ngradient(self, ctx):
        ctx.ngradient = ctx.in_values[0] / abs(ctx.in_values[0])

    def __str__(self):
        e = graph.in_edges(self)
        return 'abs({})'.format(e[0][0])
