
from numbers import Number
from cgraph.graphs import graph

def wrap_args(*args):
    newargs = []
    for n in args:               
        if isinstance(n, Number):
            from cgraph.constants import Constant
            newargs.append(Constant(n))
        else:
            newargs.append(n)
    return newargs

def nary_function(klass, *args, **kwargs):   
    c = klass(**kwargs)
    for n in wrap_args(*args):
        graph.add_edge(n, c)
    return c