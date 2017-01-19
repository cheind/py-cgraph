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
from numbers import Number
import copy
import math

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

    def compute_value(self, values):
        return values[self]

    def compute_gradient(self, values):
        return []

    def symbolic_gradient(self):
        return []

class Constant(Node):
    """Represents a constant value in an expression tree."""

    def __init__(self, value):
        super(Constant, self).__init__(nary=0)
        self.value = value

    def __str__(self):
        return str(self.value)

    def compute_value(self, values):
        return self.value

    def compute_gradient(self, values):
        return []

    def symbolic_gradient(self):
        return []

class Add(Node):
    """Binary addition of two nodes."""

    def __init__(self):
        super(Add, self).__init__(nary=2)

    def __str__(self):
        return '({} + {})'.format(str(self.children[0]), str(self.children[1]))

    def compute_value(self, values):
        return values[self.children[0]] + values[self.children[1]]
    
    def compute_gradient(self, values):
        return [1, 1]
    
    def symbolic_gradient(self):
        return [Constant(1), Constant(1)]

class Sub(Node):
    """Binary subtraction of two nodes."""

    def __init__(self):
        super(Sub, self).__init__(nary=2)

    def __str__(self):
        return '({} - {})'.format(str(self.children[0]), str(self.children[1]))

    def compute_value(self, values):
        return values[self.children[0]] - values[self.children[1]]
    
    def compute_gradient(self, values):
        return [1, -1]
    
    def symbolic_gradient(self):
        return [Constant(1), Constant(-1)]

class Mul(Node):
    """Binary multiplication of two nodes."""

    def __init__(self):
        super(Mul, self).__init__(nary=2)

    def __str__(self):
        return '({}*{})'.format(str(self.children[0]), str(self.children[1]))

    def compute_value(self, values):
        return values[self.children[0]] * values[self.children[1]]
    
    def compute_gradient(self, values):
        return [values[self.children[1]], values[self.children[0]]]

    def symbolic_gradient(self):
        return [self.children[1], self.children[0]]

def nan_on_fail(f):
    """Catches math exceptions when evaluating `f` and turns them into NAN."""
    try:
        return f()
    except (ArithmeticError, ValueError):
        return float('nan')

class Div(Node):
    """Binary division of two nodes."""

    def __init__(self):
        super(Div, self).__init__(nary=2)

    def __str__(self):
        return '({}/{})'.format(str(self.children[0]), str(self.children[1]))

    def compute_value(self, values):
        return nan_on_fail(lambda: values[self.children[0]] / values[self.children[1]])
    
    def compute_gradient(self, values):
        return [
            nan_on_fail(lambda: 1. / values[self.children[1]]), 
            nan_on_fail(lambda: -values[self.children[0]]/ values[self.children[1]]**2)
        ]
    
    def symbolic_gradient(self):
        return [
            Constant(1) / self.children[1],
            -self.children[0] / self.children[1]**2
        ]

class Logartihm(Node):
    """Natural logarithm of a node."""

    def __init__(self):
        super(Logartihm, self).__init__(nary=1)

    def __str__(self):
        return 'log({})'.format(str(self.children[0]))

    def compute_value(self, values):
        return nan_on_fail(lambda: math.log(values[self.children[0]]))

    def compute_gradient(self, values):
        return [nan_on_fail(lambda: 1. / values[self.children[0]])]

    def symbolic_gradient(self):
        return [Constant(1) / self.children[0]]


class Neg(Node):
    """Unary negation of a node."""

    def __init__(self):
        super(Neg, self).__init__(nary=1)

    def __str__(self):
        return '-{}'.format(str(self.children[0]))

    def compute_value(self, values):
        return -values[self.children[0]]

    def compute_gradient(self, values):
        return [-1]

    def symbolic_gradient(self):
        return [Constant(-1)]


class Pow(Node):
    """Binary exponentiation `x**y`."""

    def __init__(self):
        super(Pow, self).__init__(nary=2)

    def __str__(self):
        return '{}**{}'.format(str(self.children[0]), str(self.children[1]))

    def compute_value(self, values):
        return values[self.children[0]]**values[self.children[1]]

    def compute_gradient(self, values):
        return [
            values[self.children[1]] * values[self.children[0]]**(values[self.children[1]]-1),
            nan_on_fail(lambda: values[self] * math.log(values[self.children[0]]))
        ]

    def symbolic_gradient(self):
        return [
            self.children[1] * self.children[0] ** (self.children[1]-1),
            self * sym_log(self.children[0])
        ]
        

def wrap_args(func):
    """Decorator that turns plain number arguments into Constant objects."""
    def wrapped(*args, **kwargs):
        new_args = []
        for a in args:
            if isinstance(a, Number):
                a = Constant(a)
            new_args.append(a)
        return func(*new_args, **kwargs)
    return wrapped
        
@wrap_args
def sym_add(x, y):
    """Returns a new node that represents of `x+y`."""
    n = Add()
    n.children[0] = x
    n.children[1] = y
    return n

@wrap_args
def sym_sub(x, y):
    """Returns a new node that represents of `x-y`."""
    n = Sub()
    n.children[0] = x
    n.children[1] = y
    return n

@wrap_args
def sym_mul(x, y):
    """Returns a new node that represents of `x*y`."""
    n = Mul()
    n.children[0] = x
    n.children[1] = y
    return n

@wrap_args
def sym_div(x, y):
    """Returns a new node that represents of `x/y`."""
    n = Div()
    n.children[0] = x
    n.children[1] = y
    return n

@wrap_args
def sym_log(x):
    """Returns a new node that represents of `ln(x)`."""
    n = Logartihm()
    n.children[0] = x
    return n

@wrap_args
def sym_neg(x):
    """Returns a new node that represents of `-x`."""
    n = Neg()
    n.children[0] = x
    return n

@wrap_args
def sym_pow(x, y):
    """Returns a new node that represents of `x**y`."""
    n = Pow()
    n.children[0] = x
    n.children[1] = y
    return n

def sym_sum(x):
    """
    Returns a new node that represents the sum over all elements in `x`.
    Currently this is accomplished by a series of binary additions. One might
    however also consider a more efficient implementation using a new n-ary Sum node.
    """
    if len(x) == 0:
        return Constant(0)   
    
    n = x[0]
    for idx in range(1, len(x)):
        n = n + x[idx]
    return n

def postorder(node):
    """
    Yields all nodes discovered by depth-first-search in post-order starting from node.

    Note, this implementation uses a recursion. As such it has a limit on the
    how depth expression trees can become before Python raises maximum recursion depth exception.
    """
    for c in node.children:
        yield from postorder(c)
    yield node

def bfs(node, node_data):
    """
    Yields all nodes and associated data in breadth-first-search.

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
            
def values(f, fargs):
    """
    Returns a dictionary of computed values for each node in the expression tree including `f`.
    
    It is assumed by the implementation of this function that missing values
    for Symbols are given in `fargs`.
    """
    v = {}
    v.update(fargs)
    for n in postorder(f):
        if not n in v:
            v[n] = n.compute_value(v)
    return v

def value(f, fargs):
    """Shortcut for `values(f, fargs)[f]`."""
    return values(f, fargs)[f]
    
def numeric_gradient(f, fargs):
    """Computes the numerical partial derivatives of `f` with respect to all nodes using backpropagation."""
    vals = values(f, fargs)
    derivatives = defaultdict(lambda: 0)

    gen = bfs(f, 1)
    try:
        n, in_grad = next(gen)
        while True:
            derivatives[n] += in_grad
            local_grad = n.compute_gradient(vals)
            n, in_grad = gen.send([l*in_grad for l in local_grad])
    except StopIteration:
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

