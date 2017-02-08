"""CGraph - symbolic computation in Python library.

This library is the result of my efforts to understand symbolic computation of
functions factored as expression trees. In a few lines of code it shows how to
forward evaluate functions and how to perform numeric and symbolic derivatives
computations using backpropagation.

While this library is not complete (and will never be) it offers the interested
reader some insights on one way in which symbolic computation can be performed.

The code is accompanied by a series of notebooks that explain the fundamental
concepts. You can find these notebooks online at

    https://github.com/cheind/py-cgraph

Christoph Heindl, 2017
"""

import cgraph as cg
import numpy as np

from contextlib import contextmanager

_state = {
    'x': cg.Symbol('x'),
    'y': cg.Symbol('y'),
    'smoothness': 0
}
"""Tracks SDF properties. 

These properties are applied when modelling SDF functions involving functions that cannot take
parameters. Using `&` for joining two SDFs does not allow for any properties such as smoothness
parameters to be passed along. You should not modify the state directly but instead use 
context managers to adapt settings.

Elements:
    'x' : Symbol representing variable spatial `x` coordinate in SDF expressions.
    'y' : Symbol representing variable spatial `y` coordinate in SDF expressions.
    'smoothness' : Controls the smoothness when joining / intersecting signed distance functions
"""

def properties(newprops):
    """Updates global state using new properties.
    
    This method first saves the old state, then applies
    the properties to the actual state. Control is handed back
    to the caller via yield. Finally the old state is reverted.

    Method is assumed to be called from contextmanager objects.
    """
    global _state        
    try:        
        prev = _state
        _state = {**_state, **newprops}
        yield       
    finally:
        _state = prev

@contextmanager
def smoothness(s):
    """Controls the smoothness of union and intersection operations."""
    yield from properties({'smoothness':s})

@contextmanager
def transform(angle=0., offset=[0,0]):
    """Controls the origin / orientation of newly created items SDF leaves."""

    c = np.cos(angle)
    s = np.sin(angle)
    x = _state['x']
    y = _state['y']
    
    # Note, the following expressions (one for x, one for y) actually
    # use the inverse of the transform, since inversly transforming the 
    # coordinates is equivalent to 'positively' transforming the signed 
    # distance function.
    e = [
        x * c + y * s - (offset[0] * c + offset[1] * s),
        -x * s + y * c - (-offset[0] * s + offset[1] * c)
    ]
    yield from properties({'x':e[0], 'y':e[1]})


_zeroeps = np.nextafter(0, 1)

@cg.wrap_args
def sym_smax(a, b, k=32):    
    """Smooth maximum of two SDF expressions `max(a,b)`."""
    r = cg.sym_exp(k * a) + cg.sym_exp(k * b)
    return cg.sym_log(cg.sym_max(r, _zeroeps)) / k

@cg.wrap_args
def sym_smin(a, b, k=32):
    """Smooth minimum of two SDF expressions `min(a,b)`."""
    # Note that min(a,b) = -max(-a, -b)
    # http://math.stackexchange.com/questions/30843/is-there-an-analytic-approximation-to-the-minimum-function
    r = cg.sym_exp(-k * a) + cg.sym_exp(-k * b)
    return -cg.sym_log(cg.sym_max(r, _zeroeps)) / k

class SDFNode:
    """Base class for nodes in an SDF expression.

    While hierarchies of SDFNodes are similar to CGraph expression trees but are not
    inheriting from cg.Node. Instead SDFNodes hold a CGraph expression tree
    representing the actual signed distance function. This separate hierarchy is
    created to support a domain specific language for creating and manipulating
    expressions involving SDFs. For example the intersection of circle and a line
    can be written as

        s = sdf.Circle(center=[0, -0.8], radius=0.5) & sdf.Line(normal=[0.1, 1], d=-0.5)
    """

    def __init__(self, sdf):
        """Inititialize with sdf expression."""
        self.sdf = sdf
        self.F = None
        
    def __call__(self, x, y, compute_gradient=False):
        """Returns the signed distances for all pairs of x,y coordinates."""

        if self.F is None: # Lazy construction
            self.F = cg.Function(self.sdf, [_state['x'], _state['y']])

        return self.F(x, y, compute_gradient=compute_gradient)

    def __or__(self, other):
        """Union with other node."""
        return Union(self, other, k=_state.get('smoothness', 0))

    def __and__(self, other):
        """Intersection with other node."""
        return Intersection(self, other, k=_state.get('smoothness', 0))

    def __sub__(self, other):
        """Difference with other node."""
        return Difference(self, other)    

class Circle(SDFNode):
    """Represents the SDF of a circle in 2D."""

    def __init__(self, center=[0,0], radius=1):
        sdf = cg.sym_sqrt((center[0] - _state['x'])**2 + (center[1] - _state['y'])**2) - radius
        super(Circle, self).__init__(sdf)

class Halfspace(SDFNode):
    """Represents the SDF of an infinite half-space in 2D.

    The half-space is parametrized in Hessian normal form by its 
    normal vector and distance from origin.
    """

    def __init__(self, normal=[1,0], d=0):
        n =  normal / np.linalg.norm(normal)
        sdf = n[0] * _state['x'] + n[1] * _state['y'] - d
        
        super(Halfspace, self).__init__(sdf)

class Box(SDFNode):
    """Represents the SDF of a axis aligned rectangle.

    The box is parametrized by a minimum and maximum corner.
    """

    def __init__(self, minc=[-1,-1], maxc=[1,1]):

        bottom = Halfspace(normal=[0,-1], d=np.dot([0,-1], minc))
        right = Halfspace(normal=[1, 0], d=np.dot([1, 0], maxc))
        top = Halfspace(normal=[0, 1], d=np.dot([0, 1], maxc))
        left = Halfspace(normal=[-1, 0], d=np.dot([-1, 0], minc))

        box = bottom & right & top & left

        super(Box, self).__init__(box.sdf)


class Union(SDFNode):
    """Represents the union of two SDFs.

    Based on the parameter `k` the union is either peformed smoothly or hard.
    """
    def __init__(self, left, right, k=None):
        if k:
            sdf = sym_smin(left.sdf, right.sdf, k=k)            
        else:
            sdf = cg.sym_min(left.sdf, right.sdf)
        
        super(Union, self).__init__(sdf)

class Difference(SDFNode):
    """The difference between two SDFs."""

    def __init__(self, left, right):
        sdf = cg.sym_max(left.sdf, -right.sdf)
        super(Difference, self).__init__(sdf)

class Intersection(SDFNode):
    """The intersection of two SDFs.

    Based on the parameter `k` the union is either peformed smoothly or hard.
    """
    
    def __init__(self, left, right, k=None):
        if k:
            sdf = sym_smax(left.sdf, right.sdf, k=k)            
        else:
            sdf = cg.sym_max(left.sdf, right.sdf)

        super(Intersection, self).__init__(sdf)

def grid_eval(sdf, bounds=[(-2,2), (-2,2)], samples=[100j, 100j]):
    """Returns the signed distance values and gradients evaluated at corners of a regular grid."""
    x, y = np.mgrid[
        bounds[0][0]:bounds[0][1]:samples[0], 
        bounds[1][0]:bounds[1][1]:samples[1]
    ]
    d, grads = sdf(x.reshape(-1), y.reshape(-1), compute_gradient=True)
    return x, y, d.reshape(x.shape), grads.reshape(x.shape + (2,))

class GridSDF:
    """Provides fast approximate signed distance value / gradient computations.

    The performance of computing signed distance values decreases with deeper
    and more nested SDFNode hierarchies. GridSDF provides a method to make 
    signed distance values / gradients computations O(1) independent of the
    function itself. This is accomplished by rasterizing the signed distance
    function at grid corners once. When queried, distance values are simply 
    looked up in the grid data. Since positions usually don't fall
    at corners exactly, GridSDF performs a bilinear interpolation of values.
    """

    def __init__(self, sdf, bounds=[(-2,2), (-2,2)], samples=[100j, 100j]):
        x, y, d, g = grid_eval(sdf, bounds=bounds, samples=samples)

        self.d = d
        self.g = g
        self.xmin = bounds[0][0]
        self.ymin = bounds[1][0]
        self.xres = x[1, 0] - x[0, 0]
        self.yres = y[0, 1] - y[0, 0]

    def __call__(self, x, y, compute_gradient=False):
        from scipy.ndimage import map_coordinates
   
        # Map from 'world space' to grid space
        x = (np.atleast_1d(x) - self.xmin) / self.xres
        y = (np.atleast_1d(y) - self.ymin) / self.yres

        d = map_coordinates(self.d, [x, y], order=1, mode='nearest')
        gx = map_coordinates(self.g[:,:,0], [x, y], order=1, mode='nearest')
        gy = map_coordinates(self.g[:,:,1], [x, y], order=1, mode='nearest')
        
        return d, np.column_stack((gx, gy))

