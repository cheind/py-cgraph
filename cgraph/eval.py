from cgraph.graphs import graph
from cgraph.symbols import Symbol
from cgraph.context import Context


def eval(node, **kwargs):
    subgraph = graph.chain(node)     
    order = subgraph.topological_sort()

    values = {}

    for n in order:
        if graph.indegree(n) == 0 and isinstance(n, Symbol):
            assert n.name in kwargs, 'Missing input for node {}'.format(n)
            values[n] = kwargs[n.name]
        else:
            ctx = Context()
            ctx.in_values = [values[e[0]] for e in subgraph.in_edges(n)]
            n.value(ctx)
            values[n] = ctx.value
            
    return values[node]