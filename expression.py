from collections import defaultdict

class Node:

    def __init__(self, nary=0):
        self.children = [None]*nary

class Symbol(Node):

    def __init__(self, name):
        super(Symbol, self).__init__(nary=0)
        self.value = None

    def compute_value(self, values):
        return self.value

    def compute_gradient(self, values):
        return 0.

class Add(Node):

    def __init__(self):
        super(Add, self).__init__(nary=2)

    def compute_value(self, values):
        return values[self.children[0]] + values[self.children[1]]
    
    def compute_gradient(self, values):
        return [1, 1]


class Mul(Node):

    def __init__(self):
        super(Mul, self).__init__(nary=2)

    def compute_value(self, values):
        return values[self.children[0]] * values[self.children[1]]
    
    def compute_gradient(self, values):
        return [values[self.children[1]], values[self.children[0]]]

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
            
def values(node):
    v = {}
    for n in postorder(node):
        if not n in v:
            v[n] = n.compute_value(v)
    return v

def nderivatives(node):
    vals = values(node)
    derivatives = defaultdict(lambda: 0)
    derivatives[node] = 1.

    for n in bfs(node):
        d = derivatives[n]
        g = n.compute_gradient(vals)
        for idx, c in enumerate(n.children):
            derivatives[c] += g[idx] * d

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

    print(nderivatives(mul))