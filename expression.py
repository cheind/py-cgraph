from collections import defaultdict
from numbers import Number

class Node:

    def __init__(self, nary=0):
        self.children = [None]*nary

    def __repr__(self):
        return self.__str__()

    def __add__(self, other):
        return sym_add(self, other)
    
    def __sub__(self, other):
        return sym_sub(self, other)

    def __mul__(self, other):
        return sym_mul(self, other)

    def __truediv__(self, other):
        return sym_div(self, other)

class Symbol(Node):

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
    try:
        return f()
    except (ArithmeticError, ValueError):
        return float('nan')

class Div(Node):

    def __init__(self):
        super(Div, self).__init__(nary=2)

    def __str__(self):
        return '({}/{})'.format(str(self.children[0]), str(self.children[1]))

    def compute_value(self, values):
        return values[self.children[0]] / values[self.children[1]]
    
    def compute_gradient(self, values):
        return [
            nan_on_fail(lambda: 1. / values[self.children[1]]), 
            nan_on_fail(lambda: -values[self.children[0]]/ values[self.children[1]]**2)
        ]
    
    def symbolic_gradient(self):
        return [
            Constant(1) / self.children[1],
            Constant(-1) * self.children[0] / sym_sqr(self.children[1])
        ]

def wrap_args(func):
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
    n = Add()
    n.children[0] = x
    n.children[1] = y
    return n

@wrap_args
def sym_sub(x, y):
    n = Sub()
    n.children[0] = x
    n.children[1] = y
    return n

@wrap_args
def sym_mul(x, y):
    n = Mul()
    n.children[0] = x
    n.children[1] = y
    return n

@wrap_args
def sym_div(x, y):
    n = Div()
    n.children[0] = x
    n.children[1] = y
    return n

@wrap_args
def sym_sqr(x):
    return x*x

def sym_sum(x):
    if len(x) == 0:
        return Constant(0)   
    
    n = x[0]
    for idx in range(1, len(x)):
        n = n + x[idx]
    return n

def postorder(node):
    for c in node.children:
        yield from postorder(c)
    yield node

def bfs(node, node_data):
    q = [(node, node_data)]
    while q:
        t = q.pop(0)
        node_data = yield t
        for idx, c in enumerate(t[0].children):
            q.append((c, node_data[idx]))
            
def values(f, fargs):
    v = {}
    v.update(fargs)
    for n in postorder(f):
        if not n in v:
            v[n] = n.compute_value(v)
    return v

def value(f, fargs):
    return values(f, fargs)[f]
    
def numeric_gradient(f, fargs):
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
    """Decorates rule functions to match specific nodes in simplification."""

    def wrapper(func):
        def wrapped_func(node):
            if isinstance(node, klasses):
                return func(node)
            else:
                return node
        return wrapped_func
    return wrapper

def is_const(node, value=None):
    if isinstance(node, Constant):
        if value is not None:
            return node.value == value
        else:
            return True            
    return False

@applies_to(Mul)
def mul_identity_rule(node):
    if is_const(node.children[0], 1):
        return node.children[1]
    elif is_const(node.children[1], 1):
        return node.children[0]
    else:
        return node

@applies_to(Add)
def add_identity_rule(node):
    if is_const(node.children[0], 0):
        return node.children[1]
    elif is_const(node.children[1], 0):
        return node.children[0]
    else:
        return node

def eval_to_const_rule(node):
    try:
        k = value(node, {})
        return Constant(k)
    except KeyError:
        return node

import copy
def simplify(node, other_rules=None):
    """Returns a simplified version of the forward graph associated with the given node."""

    rules = [mul_identity_rule, add_identity_rule, eval_to_const_rule]
    if other_rules:
        rules.extend(other_rules)

    nodemap = {}
    for n in postorder(node):
        if isinstance(n, Symbol):
            continue

        nc = copy.copy(n)
        for i in range(len(nc.children)):
            c = nc.children[i]
            nc.children[i] = nodemap.get(c, c)
        for r in rules:
            nc = r(nc)
        nodemap[n] = nc
        
    return nodemap[node]