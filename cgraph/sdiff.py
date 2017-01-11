from collections import defaultdict

from cgraph.graphs import graph
from cgraph.constants import Constant
from cgraph.context import Context
from cgraph.helpers import arraylike

def multiplicity(edges):
    c = defaultdict(lambda: 0)
    for e in edges:
        c[e] += 1
    return c

def symbolic_sum(nodes):
    n = nodes[0]
    for idx in range(1, len(nodes)):
        n = n + nodes[idx]
    return n

def sdiff(node):
    subgraph = graph.chain(node)     
    order = subgraph.topological_sort()

    # Forward pass
    grads = {}

    for n in order:
        ctx = Context()
        ctx.in_edges = subgraph.in_edges(n)
        
        if len(ctx.in_edges) == 0:
            continue
        
        ctx.in_nodes = [e[0] for e in ctx.in_edges]
        n.sgradient(ctx)

        g = arraylike(ctx.sgradient)
        mult = multiplicity(ctx.in_edges)
        for idx, e in enumerate(ctx.in_edges):
            m = mult[e]
            grads[e] = g[idx] if m == 1 else g[idx] * m

    # Backward pass
    diffs = {}
    for n in order[::-1]:

        d = Constant(1)
        if n != node:                   
            d = symbolic_sum([grads[e] for e in subgraph.unique_out_edges(n)])

        diffs[n] = d
        
        for e in graph.unique_in_edges(n):
            grads[e] = d * grads[e]

    return diffs