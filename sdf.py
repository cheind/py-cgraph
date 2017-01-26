import cgraph as cg
  

class Min(cg.Node):
    """Minimum of multiple elements `min(x, y, z, ...)`."""

    def __init__(self, nary):
        super(Min, self).__init__(nary=nary)

    def __str__(self):
        return 'min({})'.format(','.join([str(c) for c in self.children]))

    def compute_value(self, values):
        return min([values[c] for c in self.children])

    def compute_gradient(self, values):
        vals = [values[c] for c in self.children]
        val, idx = min((val, idx) for (idx, val) in enumerate(vals))

        g = [0]*len(vals)
        g[idx] = 1.
        
        return g

@cg.wrap_args
def sym_min(*args):
    """Returns a new node that represents `min(x,y,z,...)`."""
    n = Min(nary=len(args))
    for i, a in enumerate(args):
        n.children[i] = a
    return n

@cg.wrap_args
def circle(x, y, cx=0., cy=0., r=1.):
    """Return the signed distance function for a circle."""
    return cg.sym_sqrt((cx - x)**2 + (cy - y)**2) - r

@cg.wrap_args
def union(*args):
    """Return union of multiple signed distance functions."""
    return sym_min(*args)

