import cgraph as cg
import numpy as np
  

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
def line(x, y, nx=1., ny=0., d=0):
    """Return the signed distance function for a line."""
    return nx * x + ny * y - d

@cg.wrap_args
def union(*args):
    """Return union of multiple signed distance functions."""
    return sym_min(*args)


cg.Node.__or__ = lambda self, other: union(self, other)


class Function:

    def __init__(self, f, symbols):
        self.f = f
        self.syms = [(i, s) for i, s in enumerate(symbols)]

    def __call__(self, *values, with_gradient=False):
        if len(values) == 0:
            return cg.value(self.f, {})
        
        values = [np.atleast_1d(v) for v in values]
        n = len(values[0])
        
        rv = np.empty(n)
        rg = np.empty((n, len(self.syms))) if with_gradient else None
        e = {}

        if with_gradient:
            for i in range(n):
                e = dict([(s, values[si][i]) for si, s in self.syms])
                g, v = cg.numeric_gradient(self.f, e, with_gradient=True)
                rv[i] = v[self.f]                
                rg[i][:] = [g[s] for si, s in self.syms]
            return rv, rg
        else:
            for i in range(n):
                e = dict([(s, values[si][i]) for si, s in self.syms])
                v = cg.value(self.f, e)
                rv[i] = v
            return rv