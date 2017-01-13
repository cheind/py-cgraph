from cgraph.graphs import GraphTraversal

def plot(node, filename):
    """Save the forward graph associated with given node in raw graphviz format."""
    
    import graphviz as gv
    import uuid
    t = GraphTraversal(node)
    
    g = gv.Digraph()
    g.graph_attr['rankdir'] = 'LR'

    nodemap = {}
    for n in t.forward_order():
        nodemap[n] = str(uuid.uuid4())
        g.node(nodemap[n], str(n))

    
    for n in t.forward_order():
        out_e = t.graph.out_edges(n)
        for e in out_e:
            g.edge(nodemap[e[0]], nodemap[e[1]])
    
    g.save(filename)
    