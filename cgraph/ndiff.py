from collections import defaultdict

import cgraph.graphs as graphs
from cgraph.symbols import Symbol
from cgraph.eval import Context
from cgraph.helpers import arraylike

def ndiff(node, fargs):
    t = graphs.GraphTraversal(node)

    # Forward pass
    values = {}
    grads = defaultdict(lambda: 0.)

    for n in t.forward_order():
        if t.graph.indegree(n) == 0 and isinstance(n, Symbol):
            assert n in fargs, 'Missing input for node {}'.format(n)
            values[n] = fargs[n]
        else:
            ctx = Context()
            ctx.in_edges = t.graph.in_edges(n)
            ctx.in_values = [values[e[0]] for e in ctx.in_edges]                
            
            n.value(ctx)
            n.ngradient(ctx)
            
            values[n] = ctx.value
            d = arraylike(ctx.ngradient)

            for idx, e in enumerate(ctx.in_edges):
                grads[e] += d[idx] # For duplicate edges

    # Backward pass
    diffs = {}
    for n in t.backward_order():

        d = 1.
        if n != node:                   
            d = sum([grads[e] for e in t.graph.unique_out_edges(n)])

        diffs[n] = d
        
        for e in t.graph.unique_in_edges(n):
            grads[e] *= d

    return diffs