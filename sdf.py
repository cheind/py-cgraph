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
        m = (cv[0] <= cv[1])
        ids = np.where(m)[0]

        a = np.zeros(m.shape); a[ids] = 1.
        b = np.ones(m.shape); b[ids] = 0.
        return [a,b]

class Max(cg.Node):
    """Maximum of two expressions `max(x, y)`."""

    def __init__(self):
        super(Max, self).__init__(nary=2)

    def __str__(self):
        return 'max({},{})'.format(str(self[0]), str(self[1]))

    def compute_value(self, v):
        return np.maximum(v[0], v[1])

    def compute_gradient(self, cv, value):
        m = (cv[0] >= cv[1])
        ids = np.where(m)[0]

        a = np.zeros(m.shape); a[ids] = 1.
        b = np.ones(m.shape); b[ids] = 0.
        return [a,b]

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

zeroeps = np.nextafter(0, 1)

@cg.wrap_args
def sym_smin(a, b, k=32):
    # Note that min(a,b) = -max(-a, -b)
    # http://math.stackexchange.com/questions/30843/is-there-an-analytic-approximation-to-the-minimum-function
    r = cg.sym_exp(-k * a) + cg.sym_exp(-k * b)
    return -cg.sym_log(sym_max(r, zeroeps)) / k

@cg.wrap_args
def sym_smax(a, b, k=32):    
    r = cg.sym_exp(k * a) + cg.sym_exp(k * b)
    return cg.sym_log(sym_max(r, zeroeps)) / k

class SDFNode:

    x = cg.Symbol('x')
    y = cg.Symbol('y')

    def __init__(self, sdf):
        self.sdf = sdf
        self.F = None
        
    def __call__(self, x, y, compute_gradient=False):
        """Provides function call semantics for expression tree represented by this node. """
        if self.F is None:
            self.F = cg.Function(self.sdf, [SDFNode.x, SDFNode.y])

        return self.F(x, y, compute_gradient=compute_gradient)

    def __or__(self, other):
        return Union(self, other, k=10)

    def __and__(self, other):
        return Intersection(self, other, k=10)

    def __sub__(self, other):
        return Difference(self, other)
    

class Circle(SDFNode):

    def __init__(self, center=[0,0], radius=1):
        sdf = cg.sym_sqrt((center[0] - SDFNode.x)**2 + (center[1] - SDFNode.y)**2) - radius
        super(Circle, self).__init__(sdf)

class Line(SDFNode):
    def __init__(self, normal=[1,0], d=0):
        n =  normal / np.linalg.norm(normal)
        sdf = n[0] * SDFNode.x + n[1] * SDFNode.y - d
        
        super(Line, self).__init__(sdf)

class Union(SDFNode):
    def __init__(self, left, right, k=None):
        if k == None:
            sdf = sym_min(left.sdf, right.sdf)
        else:
            sdf = sym_smin(left.sdf, right.sdf, k=k)
        
        super(Union, self).__init__(sdf)

class Difference(SDFNode):

    def __init__(self, left, right):
        sdf = sym_max(left.sdf, -right.sdf)
        super(Difference, self).__init__(sdf)

class Intersection(SDFNode):
    
    def __init__(self, left, right, k=None):
        if k is None:
            sdf = sym_max(left.sdf, right.sdf)
        else:
            sdf = sym_smax(left.sdf, right.sdf, k=k)
        super(Intersection, self).__init__(sdf)

def grid_eval(sdf, bounds=[(-2,2), (-2,2)], samples=[100j, 100j]):
    y, x = np.mgrid[
        bounds[0][0]:bounds[0][1]:samples[0], 
        bounds[1][0]:bounds[1][1]:samples[1]
    ]
    d, grads = sdf(x.reshape(-1), y.reshape(-1), compute_gradient=True)
    return x, y, d.reshape(x.shape), grads.reshape(x.shape + (2,))

class GridSDF:

    def __init__(self, sdf, bounds=[(-2,2), (-2,2)], samples=[100j, 100j]):
        x, y, d, g = grid_eval(sdf, bounds=bounds, samples=samples)

        self.d = d
        self.g = g
        self.xmin = bounds[0][0]
        self.ymin = bounds[1][0]
        self.xres = x[0, 1] - x[0, 0]
        self.yres = y[1, 0] - y[0, 0]

    def __call__(self, x, y, compute_gradient=False):
        from scipy.ndimage import map_coordinates
   
        x = (np.atleast_1d(x) - self.xmin) / self.xres
        y = (np.atleast_1d(y) - self.ymin) / self.yres


        d = map_coordinates(self.d, [y, x], order=1, mode='reflect')
        gx = map_coordinates(self.g[:,:,0], [y, x], order=1, mode='reflect')
        gy = map_coordinates(self.g[:,:,1], [y, x], order=1, mode='reflect')
        
        return d, np.column_stack((gx, gy))

