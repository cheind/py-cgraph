from collections import Iterable

def parents(start_nodes, include_start_nodes=True):
    if not isinstance(start_nodes, Iterable):
        start_nodes = [start_nodes]
    
    nodes = set()
    q = list(start_nodes)
    while q:
        n = q.pop(0)
        parents = [p for p in n.ins]
        nodes.update(parents)
        q.extend(parents)

    if include_start_nodes:
        nodes.update(start_nodes)

    return list(nodes)

def topological_sort(nodes):
    if not isinstance(nodes, Iterable):
        nodes = [nodes]

    q = []
    indegree = {}
    for n in nodes:
        indeg = len(n.ins)
        indegree[n] = indeg
        if indeg == 0:
            q.append(n)

    order = []
    while q:
        n = q.pop(0)
        order.append(n)
        for s in n.outs:
            if s not in indegree:
                continue # Only consider nodes within initial list of nodes
            indegree[s] -= 1
            if indegree[s] == 0:
                q.append(s)    

    return order

if __name__ == '__main__':

    from node import Node

    x = Node('x')
    y = Node('y')
    z = Node('z')
    a = Node('a')
    w = Node('w')

    Node.add_edge(x, z)
    Node.add_edge(y, z)
    Node.add_edge(z, a)
    Node.add_edge(x, w)

    nodes = parents(a) # all but w    
    assert set(nodes) == set([x,y,z,a]), 'All but w'
    order = topological_sort(nodes)
    assert order == [x,y,z,a] or order == [y,x,z,a], 'Failed to sort topologically'