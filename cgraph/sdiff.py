from collections import defaultdict

import cgraph.graphs as graphs
from cgraph.constants import Constant
from cgraph.eval import Context
from cgraph.helpers import arraylike
from cgraph.ops.addition import sym_sum

def multiplicity(edges):
    c = defaultdict(lambda: 0)
    for e in edges:
        c[e] += 1
    return c

def sdiff(node):
    t = graphs.GraphTraversal(node)

    # Forward pass
    grads = {}

    for n in t.forward_order():
        ctx = Context()
        ctx.in_edges = t.graph.in_edges(n)
        
        if len(ctx.in_edges) == 0:
            continue
        
        ctx.in_nodes = [e[0] for e in ctx.in_edges]
        n.sgradient(ctx)

        d = arraylike(ctx.sgradient)
        mult = multiplicity(ctx.in_edges)
        for idx, e in enumerate(ctx.in_edges):
            m = mult[e]
            grads[e] = d[idx] if m == 1 else d[idx] * m

    # Backward pass
    diffs = {}
    for n in t.backward_order():

        d = Constant(1)
        if n != node:                   
            d = sym_sum([grads[e] for e in t.graph.unique_out_edges(n)])

        diffs[n] = d
        
        for e in t.graph.unique_in_edges(n):
            grads[e] = d * grads[e]

    return diffs