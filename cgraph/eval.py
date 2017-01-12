from cgraph.graphs import graph
from cgraph.symbols import Symbol

class Context(object):
    def __init__(self):
        self.value = None
        self.ngradient = None
        self.sgradient = None
        self.cache = None


def eval(node, fargs):
    subgraph = graph.chain(node)     
    order = subgraph.topological_sort()

    values = {}

    for n in order:
        if subgraph.indegree(n) == 0 and isinstance(n, Symbol):
            assert n in fargs, 'Missing input for node {}'.format(n)
            values[n] = fargs[n]
        else:
            ctx = Context()
            ctx.in_values = [values[e[0]] for e in subgraph.in_edges(n)]
            n.value(ctx)
            values[n] = ctx.value
            
    return values[node]