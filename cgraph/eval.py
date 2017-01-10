from cgraph.graphs import graph
from cgraph.symbols import Symbol


def eval(node, **kwargs):
    subgraph = graph.chain(node)     
    order = subgraph.topological_sort()

    values = {}

    for n in order:
        if graph.indegree(n) == 0 and isinstance(n, Symbol):
            assert n.name in kwargs, 'Missing input for node {}'.format(n)
            values[n] = kwargs[n.name]
        else:
            in_edges = subgraph.in_edges(n)
            in_values = [values[e[0]] for e in in_edges]                
            f, _ = n.forward(in_values)
            values[n] = f
            
    
    return values[node]