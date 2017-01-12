
import cgraph.node as node
import cgraph.helpers as helpers

class ArithmeticNode(node.Node):

    def __add__(self, other):
        import cgraph.ops.addition as op
        return op.sym_add(self, other)

    def __sub__(self, other):
        import cgraph.ops.subtraction as op
        return op.sym_sub(self, other)
    
    def __mul__(self, other):
        import cgraph.ops.multiplication as op
        return op.sym_mul(self, other)
    
    def __truediv__(self, other):
        import cgraph.ops.division as op
        return op.sym_div(self, other)

    def __neg__(self):
        import cgraph.ops.negation as op
        return op.sym_neg(self)

    def __abs__(self):
        import cgraph.ops.absolute as op
        return op.sym_abs(self)

    def __pow__(self, other):
        import cgraph.ops.exponential as op
        return op.sym_pow(self, other)
