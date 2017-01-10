from collections import Iterable
       
class DirectedGraph:

    def __init__(self):
        self.edges = []

    def out_edges(self, n):
        return [e for e in self.edges if e[0] == n]

    def in_edges(self, n):
        return [e for e in self.edges if e[1] == n]

    def multiplicity(self, e):
        return sum([1 for x in self.edges if x == e])

    def indegree(self, n):
        return len(self.in_edges(n))

    def outdegree(self, n):
        return len(self.in_edges(n))

    def add_edge(self, src, dst):
        self.edges.append((src, dst))

    def chain(self, head):       
        nodes = set()
        nodes.add(head)
        q = [head]
        while q:
            n = q.pop(0)
            parents = [src for src,dst in self.in_edges(n)]
            nodes.update(parents)
            q.extend(parents)

        return nodes

    def topological_sort(self, nodes):
        if not isinstance(nodes, Iterable):
            nodes = [nodes]

        q = []
        indegree = {}
        for n in nodes:
            indegree[n] = self.indegree(n)
            if indegree[n] == 0:
                q.append(n)

        order = []
        while q:
            n = q.pop(0)
            order.append(n)
            for e in self.out_edges(n):
                d = e[1]                        
                if d not in indegree:
                    continue # Only consider nodes within initial list of nodes
                indegree[d] -= 1
                if indegree[d] == 0:
                    q.append(d)

        return order

graph = DirectedGraph()

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