from collections import defaultdict

from cgraph.graphs import graph
from cgraph.symbols import Symbol

def eval(node, **kwargs):
    nodes = graph.chain(node)     
    order = graph.topological_sort(nodes)

    # Forward pass
    values = {}
    grads = {}

    for n in order:
        if graph.indegree(n) == 0 and isinstance(n, Symbol):
            assert n.name in kwargs, 'Missing input for node {}'.format(n)
            values[n] = kwargs[n.name]
        else:
            # Gather sorted inputs
            in_edges = graph.in_edges(n)
            in_values = [values[e[0]] for e in in_edges]                
            # Evaluate function and gradient w.r.t inputs
            f, d = n.compute(in_values)           
            # Bookkeeping for backward pass
            values[n] = f
            for idx, e in enumerate(in_edges):
                grads[e] = d[idx]

    # Backward pass
    diffs = {}
    for n in order[::-1]:

        if n == node:
            d = 1.
        else:                
            out_edges = graph.out_edges(n)            
            d = sum([grads[e] for e in out_edges if e[1] in nodes])


        diffs[n] = d
        
        seen = set()
        for e in graph.in_edges(n):
            if not e in seen:
                grads[e] *= d
                seen.add(e)
                
    return values[node], diffs