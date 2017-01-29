import cgraph as cg
import numpy as np
  

class Min(cg.Node):
    """Minimum of two expressions `min(x, y)`."""

    def __init__(self):
        super(Min, self).__init__(nary=2)

    def __str__(self):
        return 'min({},{})'.format(str(self[0]), str(self[1]))

    def compute_value(self, v):
        return np.minimum(v[0], v[1])

    def compute_gradient(self, cv, value):
        a = np.zeros((2,) + cv[0].shape)        
        m = cv[0] <= cv[1]
        a[0, np.where(m)[0]] = 1
        a[1, np.where(np.logical_not(m))[0]] = 1
        return a

class Max(cg.Node):
    """Maximum of two expressions `max(x, y)`."""

    def __init__(self):
        super(Max, self).__init__(nary=2)

    def __str__(self):
        return 'max({},{})'.format(str(self[0]), str(self[1]))

    def compute_value(self, v):
        return np.maximum(v[0], v[1])

    def compute_gradient(self, cv, value):
        a = np.zeros((2,) + cv[0].shape)        
        
        m = cv[0] >= cv[1]
        a[0, np.where(m)[0]] = 1
        a[1, np.where(np.logical_not(m))[0]] = 1
        return a

@cg.wrap_args
def sym_min(a, b):
    """Returns a new node that represents `min(x,y)`."""
    n = Min()
    n.children[0] = a
    n.children[1] = b
    return n


@cg.wrap_args
def sym_max(a, b):
    """Returns a new node that represents `max(x,y)`."""
    n = Max()
    n.children[0] = a
    n.children[1] = b
    return n

@cg.wrap_args
def sym_smin(a, b, k=32):
    # http://www.iquilezles.org/www/articles/smin/smin.htm
    r = cg.sym_exp(-k * a) + cg.sym_exp(-k * b)
    return -cg.sym_log(r) / k

@cg.wrap_args
def circle(x, y, c=np.array([0,0]), r=1.):
    """Return the signed distance function for a circle."""
    return cg.sym_sqrt((c[0] - x)**2 + (c[1] - y)**2) - r

@cg.wrap_args
def line(x, y, n=np.array([1,0]), d=0):    
    """Return the signed distance function for a line."""
    n /= np.linalg.norm(n)
    return n[0] * x + n[1] * y - d

@cg.wrap_args
def union(a, b):
    """Return union of two signed distance functions."""
    return sym_min(a, b)

@cg.wrap_args
def subtract(a, b):
    """Return subtraction signed distance functions."""
    return sym_max(a, -b)

@cg.wrap_args
def sunion(a, b, k=5):
    """Return smooth union of two signed distance functions."""
    return sym_smin(a, b, k=k)


cg.Node.__or__ = lambda self, other: union(self, other)
