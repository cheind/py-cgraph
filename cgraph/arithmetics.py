
import cgraph.node as node
import cgraph.helpers as helpers

class ArithmeticNode(node.Node):

    def __add__(self, other):
        import cgraph.ops.addition as add
        return helpers.nary_function(add.Add, self, other)

    def __sub__(self, other):
        import cgraph.ops.subtraction as sub
        return helpers.nary_function(sub.Sub, self, other)
    
    def __mul__(self, other):
        import cgraph.ops.multiplication as mul
        return helpers.nary_function(mul.Mul, self, other)
    
    def __truediv__(self, other):
        import cgraph.ops.division as div
        return helpers.nary_function(div.Div, self, other)

    def __neg__(self):
        import cgraph.ops.negation as neg
        return helpers.nary_function(neg.Neg, self)

    def __abs__(self):
        import cgraph.ops.absolute as abso
        return helpers.nary_function(abso.Abs, self)