import cgraph as cg

class Min(cg.Node):
    """Minimum `min(x, y)`."""

    def __init__(self):
        super(Min, self).__init__(nary=2)

    def __str__(self):
        return 'min({},{})'.format(str(self.children[0]), str(self.children[0]))

    def compute_value(self, values):
        return min(values[self.children[0]], values[self.children[1]])

    def compute_gradient(self, values):
        if values[self.children[0]] <= values[self.children[1]]:
            return [1, 0]
        else:
            return [0, 1]

@cg.wrap_args
def sym_min(x,y):
    """Returns a new node that represents `min(x,y)`."""
    n = Min()
    n.children[0] = x
    n.children[1] = y
    return n

@cg.wrap_args
def circle(x, y, cx=0., cy=0., r=1.):
    """Return the signed distance function for a circle."""
    return cg.sym_sqrt((cx - x)**2 + (cy - y)**2) - r

def union(x, y):
    return sym_min(x, y)
