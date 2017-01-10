from collections import defaultdict
from cgraph import graph
from cgraph.symbols import Symbol

def eval(node, **kwargs):
    nodes = set(graph.parents(node))        
    order = graph.topological_sort(nodes)

    # Forward pass
    values = {}
    grads = defaultdict(dict)

    for n in order:
        if n.in_degree == 0 and isinstance(n, Symbol):
            assert n.name in kwargs, 'Missing input for node {}'.format(n)
            values[n] = kwargs[n.name]
        else:
            # Gather sorted inputs
            in_values = [values[i] for i in n.ins]                
            # Evaluate function and gradient w.r.t inputs
            f, d = n.compute(in_values)           
            # Bookkeeping for backward pass
            values[n] = f
            for idx, i in enumerate(n.ins):                    
                grads[i][n] = d[idx]

    # Backward pass
    diffs = {}
    for n in order[::-1]:

        if n == node:
            d = 1.
        else:                
            d = sum([v for v in grads[n].values()])

        diffs[n] = d
            
        for i,m in n.ins_with_multiplicity().items():
            grads[i][n] *= (d * m)

    return values[node], diffs