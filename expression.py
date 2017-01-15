from collections import defaultdict

class Node:

    def __init__(self, nary=0):
        self.children = [None]*nary

    def __repr__(self):
        return self.__str__()

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
    print(symbolic_derivatives(symbolic_derivatives(mul)[x]))