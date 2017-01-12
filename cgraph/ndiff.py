from collections import defaultdict

from cgraph.graphs import graph
from cgraph.symbols import Symbol
from cgraph.eval import Context
from cgraph.helpers import arraylike

def ndiff(node, fargs):
    subgraph = graph.chain(node)     
    order = subgraph.topological_sort()

    # Forward pass
    values = {}
    grads = defaultdict(lambda: 0.)

    for n in order:
        if subgraph.indegree(n) == 0 and isinstance(n, Symbol):
            assert n in fargs, 'Missing input for node {}'.format(n)
            values[n] = fargs[n]
        else:
            ctx = Context()
            ctx.in_edges = subgraph.in_edges(n)
            ctx.in_values = [values[e[0]] for e in ctx.in_edges]                
            
            n.value(ctx)
            n.ngradient(ctx)
            
            values[n] = ctx.value
            g = arraylike(ctx.ngradient)

            for idx, e in enumerate(ctx.in_edges):
                grads[e] += g[idx] # For duplicate edges

    # Backward pass
    diffs = {}
    for n in order[::-1]:

        d = 1.
        if n != node:                   
            d = sum([grads[e] for e in subgraph.unique_out_edges(n)])

        diffs[n] = d
        
        for e in graph.unique_in_edges(n):
            grads[e] *= d

    return diffs