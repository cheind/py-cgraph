from collections import defaultdict
from numbers import Number

class Node:

    def __init__(self, nary=0):
        self.children = [None]*nary

    def __repr__(self):
        return self.__str__()

    def __add__(self, other):
        return sym_add(self, other)

    def __mul__(self, other):
        return sym_mul(self, other)

class Symbol(Node):

    def __init__(self, name):
        super(Symbol, self).__init__(nary=0)
        self.name = name
        self.value = None

    def __str__(self):
        return self.name

    def compute_value(self, values):
        return self.value

    def compute_gradient(self, values):
        return None

    def symbolic_gradient(self):
        return None

class Constant(Node):

    def __init__(self, value):
        super(Constant, self).__init__(nary=0)
        self.value = value

    def __str__(self):
        return str(self.value)

    def compute_value(self, values):
        return self.value

    def compute_gradient(self, values):
        return None

    def symbolic_gradient(self):
        return None

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
def sym_mul(x, y):
    n = Mul()
    n.children[0] = x
    n.children[1] = y
    return n


def postorder(node):
    for c in node.children:
        yield from postorder(c)
    yield node

def bfs(node):
    q = [node]
    while q:
        n = q.pop(0)
        yield n
        for c in n.children:
            q.append(c)
            
def numeric_values(node):
    v = {}
    for n in postorder(node):
        if not n in v:
            v[n] = n.compute_value(v)
    return v

def numeric_derivatives(node):
    vals = numeric_values(node)
    derivatives = defaultdict(lambda: 0)
    derivatives[node] = 1.

    for n in bfs(node):
        d = derivatives[n]
        g = n.compute_gradient(vals)
        for idx, c in enumerate(n.children):
            derivatives[c] += g[idx] * d

    return derivatives

def symbolic_derivatives(node):
    derivatives = defaultdict(lambda: Constant(0))
    derivatives[node] = Constant(1)

    for n in bfs(node):
        d = derivatives[n]
        g = n.symbolic_gradient()
        for idx, c in enumerate(n.children):
            m = Mul()
            m.children[0] = g[idx]
            m.children[1] = d

            a = Add()
            a.children[0] = derivatives[c]
            a.children[1] = m

            derivatives[c] = a

    return derivatives

def symbolic_derivatives2(node):
    derivatives = defaultdict(lambda: Constant(0))
    derivatives[node] = Constant(1)

    for n in bfs(node):
        d = derivatives[n]
        g = n.symbolic_gradient()
        for idx, c in enumerate(n.children):
            derivatives[c] = derivatives[c] + (g[idx] * d)

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

import copy
def simplify(node, other_rules=None):
    """Returns a simplified version of the forward graph associated with the given node."""

    rules = [mul_identity_rule, add_identity_rule]
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

if __name__=='__main__':

    x = Symbol('x')
    y = Symbol('y')
    add = Add()
    add.children[0] = x
    add.children[1] = y

    mul = Mul()
    mul.children[0] = add
    mul.children[1] = x

    x.value = 3
    y.value = 2

    print(numeric_derivatives(mul))
    print(symbolic_derivatives(mul))
    print(symbolic_derivatives2(mul))
    print('simplification')
    print(simplify(symbolic_derivatives2(mul)[x]))
    #print(symbolic_derivatives(symbolic_derivatives(mul)[x]))