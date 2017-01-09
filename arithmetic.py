
from node import Node
from numbers import Number

def wrap_args(func):
    """Decorator to convert wrap arguments into Nodes when required."""
    def wrapper(*args):
        newargs = []
        for n in args:               
            if isinstance(n, Number):
                newargs.append(Constant(n))
            else:
                newargs.append(n)
        
        return func(*newargs)
    return wrapper

class ArithmeticNode(Node):

    @wrap_args
    def __add__(self, other):
        return Node.nary_function(Add, self, other)

    @wrap_args
    def __sub__(self, other):
        return Node.nary_function(Sub, self, other)
    
    @wrap_args
    def __mul__(self, other):
        return Node.nary_function(Mul, self, other)
    
    @wrap_args
    def __truediv__(self, other):
        return Node.nary_function(Div, self, other)

class Add(ArithmeticNode):
    
    def compute(self, inputs):
        return inputs[0] + inputs[1], [1., 1.]

    def __str__(self):
        return '({} + {})'.format(self.ins[0], self.ins[1])

class Sub(ArithmeticNode):
    
    def compute(self, inputs):
        return inputs[0] - inputs[1], [1., -1.]

    def __str__(self):
        return '({} - {})'.format(self.ins[0], self.ins[1])

class Mul(ArithmeticNode):

    def compute(self, inputs):
        return inputs[0] * inputs[1], [inputs[1], inputs[0]]

    def __str__(self):
        return '({}*{})'.format(self.ins[0], self.ins[1])

class Div(ArithmeticNode):

    def compute(self, inputs):
        return inputs[0] / inputs[1], [1. / inputs[1], -inputs[0]/inputs[1]**2]

    def __str__(self):
        return '({}/{})'.format(self.ins[0], self.ins[1])

class Constant(ArithmeticNode):
    def __init__(self, value):
        super(Constant, self).__init__()
        self.value = value

    def __str__(self):
        return str(self.value)

    """ 
    Constants are wrapped on the fly, so many such
    objects might be generated. We cannot hash those
    constants by value, as we are not obtaining a global graph
    structure. Therefore, we would just lose information.
    def __hash__(self):
        return hash(self.value)            
    
    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.value == other.value      
        else:
            return False
    """

    def compute(self, inputs):
        return self.value, [0.]

class Symbol(ArithmeticNode):
    def __init__(self, name):
        super(Symbol, self).__init__(input_required=True)
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
        
    def compute(self, inputs):
        pass


if __name__ == '__main__':

    x = Symbol('x')
    y = Symbol('y')
    z = Symbol('z')

    f = (x * y + 3) / (z - 2)
    v,d = f.eval(x=3, y=4, z=4)

    print('f {}'.format(v))
    print('df/dx {}'.format(d[x]))
    print('df/dy {}'.format(d[y]))
    print('df/dz {}'.format(d[z]))

    k = x*3-y
    v,d = k.eval(x=3, y=4)
    print(v)
    print(d)

    m = f / k
    v,d = m.eval(x=3, y=4, z=4)
    print(v)
    print(d)