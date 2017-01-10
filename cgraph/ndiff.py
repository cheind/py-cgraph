from collections import defaultdict

from cgraph.graphs import graph
from cgraph.symbols import Symbol

def ndiff(node, **kwargs):
    subgraph = graph.chain(node)     
    order = subgraph.topological_sort()

    # Forward pass
    values = {}
    grads = defaultdict(lambda: 0.)

    for n in order:
        if graph.indegree(n) == 0 and isinstance(n, Symbol):
            assert n.name in kwargs, 'Missing input for node {}'.format(n)
            values[n] = kwargs[n.name]
        else:
            # Gather sorted inputs
            in_edges = subgraph.in_edges(n)
            in_values = [values[e[0]] for e in in_edges]                
            # Evaluate function and gradient w.r.t inputs
            f, cache = n.forward(in_values)
            g = n.ngradient(cache)
            values[n] = f
            for idx, e in enumerate(in_edges):
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