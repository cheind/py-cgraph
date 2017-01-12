
import cgraph.graphs as graphs
from cgraph.symbols import Symbol

class Context(object):
    def __init__(self):
        self.value = None
        self.ngradient = None
        self.sgradient = None
        self.cache = None

def eval(node, fargs):
    t = graphs.GraphTraversal(node)

    values = {}

    for n in t.forward_order():
        if t.graph.indegree(n) == 0 and isinstance(n, Symbol):
            assert n in fargs, 'Missing input for node {}'.format(n)
            values[n] = fargs[n]
        else:
            ctx = Context()
            ctx.in_values = [values[e[0]] for e in t.graph.in_edges(n)]
            n.value(ctx)
            values[n] = ctx.value
            
    return values[node]