from cgraph.arithmetics import ArithmeticNode
from cgraph.constants import Constant
from cgraph.graphs import graph

class Add(ArithmeticNode):

    def forward(self, inputs):
        return inputs[0] + inputs[1], None

    def ngradient(self, cache):
        return [1, 1]

    def sgradient(self, inputs):
        return [Constant(1), Constant(1)]

    def __str__(self):
        e = graph.in_edges(self)
        return '({} + {})'.format(e[0][0], e[1][0])

class Sub(ArithmeticNode):

    def forward(self, inputs):
        return inputs[0] - inputs[1], None

    def ngradient(self, cache):
        return [1, -1]

    def sgradient(self, inputs):
        return [Constant(1), Constant(-1)]

    def __str__(self):
        e = graph.in_edges(self)
        return '({} - {})'.format(e[0][0], e[1][0])

class Mul(ArithmeticNode):

    def forward(self, inputs):
        return inputs[0] * inputs[1], inputs

    def ngradient(self, cache):
        return [cache[1], cache[0]]

    def sgradient(self, inputs):
        return [inputs[1], inputs[0]]

    def __str__(self):
        e = graph.in_edges(self)
        return '({}*{})'.format(e[0][0], e[1][0])

class Div(ArithmeticNode):

    def forward(self, inputs):
        return inputs[0] / inputs[1], inputs

    def ngradient(self, cache):
        return [1. / cache[1], -cache[0]/cache[1]**2]

    def __str__(self):
        e = graph.in_edges(self)
        return '({}/{})'.format(e[0][0], e[1][0])

class Neg(ArithmeticNode):

    def forward(self, inputs):
        return -inputs[0], None

    def ngradient(self, cache):
        return [-1]

    def sgradient(self, inputs):
        return [Constant(-1)]

    def __str__(self):
        e = graph.in_edges(self)
        return '-{}'.format(e[0][0])


class Abs(ArithmeticNode):

    def forward(self, inputs):
        return abs(inputs[0]), inputs

    def ngradient(self, cache):
        return [cache[0] / abs(cache[0])]

    def __str__(self):
        e = graph.in_edges(self)
        return 'abs({})'.format(e[0][0])
