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

from collections import defaultdict
from collections import Iterable
from numbers import Number
import copy
import math

import numpy as np

class Node:
    """A base class for operations, symbols and constants in an expression tree."""

    def __init__(self, nary=0):
        self.children = [None]*nary

    def __repr__(self):
        return self.__str__()

    def __add__(self, other):
        return sym_add(self, other)

    def __radd__(self, other):
        return sym_add(other, self)
    
    def __sub__(self, other):
        return sym_sub(self, other)
    
    def __rsub__(self, other):
        return sym_sub(other, self)
    
    def __mul__(self, other):
        return sym_mul(self, other)

    def __rmul__(self, other):
        return sym_mul(other, self)    

    def __truediv__(self, other):
        return sym_div(self, other)

    def __rtruediv__(self, other):
        return sym_div(other, self)

    def __neg__(self):
        return sym_neg(self)

    def __pow__(self, other):
        return sym_pow(self, other)
    
    def __getitem__(self, key):
        return self.children[key]

    def compute_value(self, cv):
        """Return the node's value computed from the values of children given as array in `cv`."""
        raise NotImplementedError()

    def compute_gradient(self, cv, value):
        """Return the node's numeric gradient evaluated."""
        raise NotImplementedError()

    def symbolic_gradient(self):
        raise NotImplementedError()

    def child_values(self, values):
        return [values[c] for c in self.children]  

class Symbol(Node):
    """
    Represents a terminal node that might be associated with a scalar value.    
    Symbols are uniquely determined by their name.
    """

    def __init__(self, name):
        super(Symbol, self).__init__(nary=0)
        self.name = name

    def __str__(self):
        return self.name

    def __hash__(self):
        return hash(self.name)            
    
    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.name == other.name      
        else:
            return False

    def compute_gradient(self, cv, v):
        return np.ones(1)

    def symbolic_gradient(self):
        return []

class Constant(Node):
    """Represents a constant value in an expression tree."""

    def __init__(self, value):
        super(Constant, self).__init__(nary=0)
        self.value = np.atleast_1d(value)

    def __str__(self):
        return str(self.value)

    def compute_value(self, values):
        return self.value

    def compute_gradient(self, cv, value):
        return np.ones(1)

    def symbolic_gradient(self):
        return []

class Add(Node):
    """Binary addition of two nodes."""

    def __init__(self):
        super(Add, self).__init__(nary=2)

    def __str__(self):
        return '({} + {})'.format(str(self[0]), str(self[1]))

    def compute_value(self, cv):
        return cv[0] + cv[1]
    
    def compute_gradient(self, cv, value):
        return [np.ones(cv[0].shape), np.ones(cv[1].shape)]
    
    def symbolic_gradient(self):
        return [Constant(1), Constant(1)]

class Sum(Node):
    """N-ary summation of nodes on a single level."""

    def __init__(self, n):
        assert n > 0, "Sum requires at least one child node"
        super(Sum, self).__init__(nary=n)

    def __str__(self):
        return '({})'.format(' + '.join([str(c) for c in self.children]))        

    def compute_value(self, cv):
        return np.sum(cv, axis=0)
    
    def compute_gradient(self, cv, value):
        return [np.ones(v.shape) for v in cv]
        
    def symbolic_gradient(self):
        return [Constant(1)]*len(self.children)

class Sub(Node):
    """Binary subtraction of two nodes."""

    def __init__(self):
        super(Sub, self).__init__(nary=2)

    def __str__(self):
        return '({} - {})'.format(str(self[0]), str(self[1]))

    def compute_value(self, cv):
        return cv[0] - cv[1]
    
    def compute_gradient(self, cv, value):
        return [np.ones(cv[0].shape), -np.ones(cv[1].shape)]
    
    def symbolic_gradient(self):
        return [Constant(1), Constant(-1)]

class Mul(Node):
    """Binary multiplication of two nodes."""

    def __init__(self):
        super(Mul, self).__init__(nary=2)

    def __str__(self):
        return '({}*{})'.format(str(self[0]), str(self[1]))

    def compute_value(self, cv):
        return cv[0] * cv[1]
    
    def compute_gradient(self, cv, value):
        return [cv[1], cv[0]]

    def symbolic_gradient(self):
        return [self[1], self[0]]

class Div(Node):
    """Binary division of two nodes."""

    def __init__(self):
        super(Div, self).__init__(nary=2)

    def __str__(self):
        return '({}/{})'.format(str(self[0]), str(self[1]))

    def compute_value(self, cv):
        return cv[0] / cv[1]
    
    def compute_gradient(self, cv, value):
        return [1. / cv[1], -cv[0] / cv[1]**2]
    
    def symbolic_gradient(self):
        return [
            Constant(1) / self[1],
            -self[0] / self[1]**2
        ]

class Logarithm(Node):
    """Natural logarithm of a node."""

    def __init__(self):
        super(Logarithm, self).__init__(nary=1)

    def __str__(self):
        return 'log({})'.format(str(self[0]))

    def compute_value(self, cv):
        return np.log(cv[0])

    def compute_gradient(self, cv, value):
        return [1./ cv[0]]

    def symbolic_gradient(self):
        return [Constant(1) / self[0]]


class Neg(Node):
    """Unary negation of a node."""

    def __init__(self):
        super(Neg, self).__init__(nary=1)

    def __str__(self):
        return '-{}'.format(str(self[0]))

    def compute_value(self, cv):
        return -cv[0]

    def compute_gradient(self, cv, value):
        return [-np.ones(cv[0].shape)]

    def symbolic_gradient(self):
        return [Constant(-1)]


class Pow(Node):
    """Binary exponentiation `x**y`."""

    def __init__(self):
        super(Pow, self).__init__(nary=2)

    def __str__(self):
        return '{}**{}'.format(str(self[0]), str(self[1]))

    def compute_value(self, cv):
        return cv[0]**cv[1]

    def compute_gradient(self, cv, value):
        return [
            cv[1] * cv[0]**(cv[1]-1), 
            value * np.log(cv[0])
        ]

    def symbolic_gradient(self):
        return [
            self[1] * self[0] ** (self[1]-1),
            self * sym_log(self[0])
        ]

class Exp(Node):
    """Base-e exponential function of x `e**x`."""

    def __init__(self):
        super(Exp, self).__init__(nary=1)

    def __str__(self):
        return 'exp({})'.format(str(self[0]))

    def compute_value(self, v):
        return np.exp(v[0])

    def compute_gradient(self, cv, value):
        return [value]
        
    def symbolic_gradient(self):
        return [self]

class Sqrt(Node):
    """Square root of `x`."""

    def __init__(self):
        super(Sqrt, self).__init__(nary=1)

    def __str__(self):
        return 'sqrt({})'.format(str(self[0]))

    def compute_value(self, v):
        return np.sqrt(v[0])

    def compute_gradient(self, cv, value):
        return [1. / (2 * value)]

    def symbolic_gradient(self):
        return [ Constant(1) / (Constant(2) * self)]  

class Min(Node):
    """Minimum of two expressions `min(x, y)`.
    
    Symbolic gradient is not yet implemented as it requires a piecewise
    construct that isn't provided by cgraph.
    """

    def __init__(self):
        super(Min, self).__init__(nary=2)

    def __str__(self):
        return 'min({},{})'.format(str(self[0]), str(self[1]))

    def compute_value(self, v):
        return np.minimum(v[0], v[1])

    def compute_gradient(self, cv, value):
        # Gradient is 1 for whatever value is less, other one is zero.
        m = (cv[0] <= cv[1])
        ids = np.where(m)[0]

        a = np.zeros(m.shape); a[ids] = 1.
        b = np.ones(m.shape); b[ids] = 0.
        return [a,b]

class Max(Node):
    """Maximum of two expressions `max(x, y)`.
    
    Symbolic gradient is not yet implemented as it requires a piecewise
    construct that isn't provided by cgraph.
    """

    def __init__(self):
        super(Max, self).__init__(nary=2)

    def __str__(self):
        return 'max({},{})'.format(str(self[0]), str(self[1]))

    def compute_value(self, v):
        return np.maximum(v[0], v[1])

    def compute_gradient(self, cv, value):
        # Gradient is 1 for whatever value is greater, other one is zero.
        m = (cv[0] >= cv[1])
        ids = np.where(m)[0]

        a = np.zeros(m.shape); a[ids] = 1.
        b = np.ones(m.shape); b[ids] = 0.
        return [a,b]

class Sin(Node):
    """Sinus of expression `sin(x)`."""

    def __init__(self):
        super(Sin, self).__init__(nary=1)

    def __str__(self):
        return 'sin({})'.format(str(self[0]))

    def compute_value(self, cv):
        return np.sin(cv[0])

    def compute_gradient(self, cv, value):
        return [np.cos(cv[0])]

    def symbolic_gradient(self):
        return [sym_cos(self[0])]  

class Cos(Node):
    """Cosine of expression `cos(x)`."""

    def __init__(self):
        super(Cos, self).__init__(nary=1)

    def __str__(self):
        return 'cos({})'.format(str(self[0]))

    def compute_value(self, cv):
        return np.cos(cv[0])

    def compute_gradient(self, cv, value):
        return [-np.sin(cv[0])]

    def symbolic_gradient(self):
        return [-sym_sin(self[0])]  

def wrap_number(n):
    """Wraps a plain number as Constant object."""
    if isinstance(n, Number):
        n = Constant(n)
    return n    

def wrap_args(func):
    """Decorator that turns plain number arguments into Constant objects."""
    def wrapped(*args, **kwargs):
        new_args = []
        for a in args:
            new_args.append(wrap_number(a))
        return func(*new_args, **kwargs)
    return wrapped
        
@wrap_args
def sym_add(x, y):
    """Returns a new node representing `x+y`."""
    n = Add()
    n.children[0] = x
    n.children[1] = y
    return n

@wrap_args
def sym_sub(x, y):
    """Returns a new node representing `x-y`."""
    n = Sub()
    n.children[0] = x
    n.children[1] = y
    return n

@wrap_args
def sym_mul(x, y):
    """Returns a new node representing `x*y`."""
    n = Mul()
    n.children[0] = x
    n.children[1] = y
    return n

@wrap_args
def sym_div(x, y):
    """Returns a new node representing `x/y`."""
    n = Div()
    n.children[0] = x
    n.children[1] = y
    return n

@wrap_args
def sym_log(x):
    """Returns a new node representing `ln(x)`."""
    n = Logarithm()
    n.children[0] = x
    return n

@wrap_args
def sym_neg(x):
    """Returns a new node representing `-x`."""
    n = Neg()
    n.children[0] = x
    return n

@wrap_args
def sym_pow(x, y):
    """Returns a new node representing `x**y`."""
    n = Pow()
    n.children[0] = x
    n.children[1] = y
    return n

@wrap_args
def sym_exp(x):
    """Returns a new node representing `e**x`."""
    n = Exp()
    n.children[0] = x
    return n
    
@wrap_args
def sym_sqrt(x):
    """Returns a new node representing `sqrt(x)`."""
    n = Sqrt()
    n.children[0] = x
    return n

@wrap_args
def sym_min(a, b):
    """Returns a new node representing `min(x,y)`."""
    n = Min()
    n.children[0] = a
    n.children[1] = b
    return n

@wrap_args
def sym_max(a, b):
    """Returns a new node representing `max(x,y)`."""
    n = Max()
    n.children[0] = a
    n.children[1] = b
    return n

@wrap_args
def sym_cos(a):
    """Returns a new node representing `cos(x)`."""
    n = Cos()
    n.children[0] = a
    return n

@wrap_args
def sym_sin(a):
    """Returns a new node representing `sin(x)`."""
    n = Sin()
    n.children[0] = a
    return n

def sym_sum(x):
    """Returns a new node that represents the sum over all elements in `x`.

    Instead of using a sequence binary `Add` that would generate very deep expression tree
    for large number of elements, we use the more efficient `Sum` that is capable of
    doing the same with a single node.
    """
    if not isinstance(x, Iterable):
        raise ValueError('Not iterable')

    if len(x) == 0:
        return Constant(0) 

    n = Sum(n=len(x))
    for idx, e in enumerate(x):
        n.children[idx] = wrap_number(e)
    return n

def postorder(node):
    """Yields all nodes discovered by depth-first-search in post-order starting from node.

    Note, this implementation uses a recursion. As such it has a limit on the
    how depth expression trees can become before Python raises maximum recursion depth exception.
    """
    for c in node.children:
        yield from postorder(c)
    yield node

def bfs(node, node_data):
    """Yields all nodes and associated data in breadth-first-search.

    Each node will be attached a node data. It is expected by this
    implementation that the caller returns (generator.send) an array
    of node_data (one for each child) for the current node processed.
    """
    q = [(node, node_data)]
    while q:
        t = q.pop(0)
        node_data = yield t
        for idx, c in enumerate(t[0].children):
            q.append((c, node_data[idx]))

def numpyify(fargs):
    """Turns each value in the given dict into a numpy array."""
    tuples = [(k, np.atleast_1d(v)) for k, v in fargs.items()]
    return dict(tuples)
            
def values(f, fargs):
    """Returns a dictionary of computed values for each node in the expression tree including `f`.
    
    It is assumed by the implementation of this function that missing values
    for Symbols are given in `fargs`.
    """

    fargs = numpyify(fargs)
    
    v = {}    
    v.update(fargs)
    
    for n in postorder(f):
        if (not n in v) and (not isinstance(n, Symbol)):
            cvalues = n.child_values(v)
            v[n] = n.compute_value(cvalues)

    return v

def value(f, fargs):
    """Shortcut for `values(f, fargs)[f]`."""
    return values(f, fargs)[f]
    
def numeric_gradient(f, fargs, return_all_values=False, return_value=False):
    """Computes the numerical partial derivatives of `f` with respect to all nodes using backpropagation."""
    
    vals = values(f, fargs)
    derivatives = defaultdict(lambda : 0.)
    
    gen = bfs(f, 1)
    try:
        n, in_grad = next(gen)
        while True:
            derivatives[n] += in_grad
            cvalues = n.child_values(vals)
            g = n.compute_gradient(cvalues, vals[n])
            n, in_grad = gen.send([gi * in_grad for gi in g])
    except StopIteration:
        if return_all_values:
            return derivatives, vals
        elif return_value:
            return derivatives, vals[f]
        else:
            return derivatives

def symbolic_gradient(f):
    """Computes the symbolic partial derivatives of `f` with respect to all nodes using backpropagation."""
    derivatives = defaultdict(lambda: Constant(0))
    gen = bfs(f, Constant(1))
    try:
        n, in_grad = next(gen) # Need to use edge info when expressions are reused!
        while True:
            derivatives[n] = derivatives[n] + in_grad
            local_grad = n.symbolic_gradient()
            n, in_grad = gen.send([l * in_grad for l in local_grad])
    except StopIteration:
        return derivatives



class Function:
    """Wraps an expression tree to support function like call syntax.
    
    It is often convenient to evaluate an expression tree using positional
    arguments for symbols. That is instead of `cg.value(f, {x:2, y:3})` one
    intends to write `f(2,3)`. When computing gradients it is often handy group
    gradients in a single numpy array instead of having to lookup individual 
    components in an dictionary. That is, instead of

        g = cg.numeric_gradient(f, {x:[2,3,3], y:[3,4,4]})
        g[x] # array of three elements representing df/dx for each value
        g[y] # array of three elements representing df/dy for each value

    With Function you can write

        F = Function(f, [x, y])
        g, v = F([2,3,3], [3,4,4], compute_gradient=True) # g = gradients, v = function value
        g.shape # 3x2 array of gradients. One gradient per row.    
    """

    def __init__(self, f, symbols):
        self.f = f
        self.syms = [(i, s) for i, s in enumerate(symbols)]

    def __call__(self, *values, compute_gradient=False):

        fargs = dict([(s, values[si]) for si, s in self.syms])
        if compute_gradient:
            g, v = numeric_gradient(self.f, fargs, return_value=True)
            # Merge gradient directions
            g = np.hstack([g[s].reshape(-1, 1) for i,s in self.syms])
            return v, g
        else:
            v = value(self.f, fargs)
            return v

def applies_to(*klasses):
    """Decorates functions to match specific nodes only in rule based expression simplification."""
    def wrapper(func):
        def wrapped_func(node):
            if isinstance(node, klasses):
                return func(node)
            else:
                return node
        return wrapped_func
    return wrapper

def is_const(node, value=None):
    """Returns true when the node is Constant and matched `value`."""
    if isinstance(node, Constant):
        if value is not None:
            return node.value == value
        else:
            return True            
    return False

@applies_to(Mul)
def mul_identity_rule(node):
    """Simplifies `x*1` to `x`."""

    if is_const(node.children[0], 1):
        return node.children[1]
    elif is_const(node.children[1], 1):
        return node.children[0]
    else:
        return node

@applies_to(Add)
def add_identity_rule(node):
    """Simplifies `x+0` to `x`."""

    if is_const(node.children[0], 0):
        return node.children[1]
    elif is_const(node.children[1], 0):
        return node.children[0]
    else:
        return node

def eval_to_const_rule(node):
    """Simplifies every expression made of Constants only to a single Constant."""
    try:
        k = value(node, {})
        return Constant(k)
    except KeyError:
        return node


"""Default simplification rules. Add more if needed."""
simplification_rules = [    
    mul_identity_rule, 
    add_identity_rule, 
    eval_to_const_rule
]

def simplify(node):
    """Returns a simplified version of the expression tree associated with `node`."""
    nodemap = {}
    for n in postorder(node):
        if isinstance(n, Symbol):
            continue

        nc = copy.copy(n)
        for i in range(len(nc.children)):
            c = nc.children[i]
            nc.children[i] = nodemap.get(c, c)
        for r in simplification_rules:
            nc = r(nc)
        nodemap[n] = nc
        
    return nodemap[node]

def simplify_all(nodes):
    """Returns simplified expression trees for all nodes in the given collection."""
    if isinstance(nodes, (defaultdict, dict)):
        result = copy.copy(nodes)
        for k, v in nodes.items():
            result[k] = simplify(v)
        return result
    elif isinstance(nodes, (list, tuple)):
        result = []
        for n in nodes:
            result.append(simplify(n))
        return result

