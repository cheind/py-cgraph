from collections import Iterable
       
class DirectedGraph:

    def __init__(self):
        self.edges = []
        self.nodes = set()

    def out_edges(self, n):
        return [e for e in self.edges if e[0] == n]

    def in_edges(self, n):
        return [e for e in self.edges if e[1] == n]

    def unique_out_edges(self, n):
        seen = set()
        for e in self.out_edges(n):
            if not e in seen:
                seen.add(e)
                yield e                
    
    def unique_in_edges(self, n):
        seen = set()
        for e in self.in_edges(n):
            if not e in seen:
                seen.add(e)
                yield e

    def multiplicity(self, e):
        return sum([1 for x in self.edges if x == e])

    def indegree(self, n):
        return len(self.in_edges(n))

    def outdegree(self, n):
        return len(self.in_edges(n))

    def add_edge(self, src, dst):
        self.edges.append((src, dst))
        self.nodes.add(src)
        self.nodes.add(dst)
    
    def add_edges(self, edges):
        for e in edges:
            self.add_edge(e[0], e[1])
    
    def add_node(self, src):
        self.nodes.add(src)

    def chain(self, head):  
        g = DirectedGraph()
        g.add_node(head)
        
        seen = set()
        q = [head]        
        while q:
            n = q.pop(0)
            if not n in seen:
                in_edges = self.in_edges(n)            
                g.add_edges(in_edges)
                q.extend([e[0] for e in in_edges])
                seen.add(n)

        return g

    def topological_sort(self):
        q = []
        indegree = {}
        for n in self.nodes:
            indegree[n] = self.indegree(n)
            if indegree[n] == 0:
                q.append(n)

        order = []
        while q:
            n = q.pop(0)
            order.append(n)
            for e in self.out_edges(n):
                d = e[1]                        
                indegree[d] -= 1
                if indegree[d] == 0:
                    q.append(d)

        return order

graph = DirectedGraph()


class GraphTraversal:

    def __init__(self, node):
        self.graph = graph.chain(node)     
        self.order = self.graph.topological_sort()

    def forward_order(self):
        for n in self.order:
            yield n

    def backward_order(self):
        for n in self.order[::-1]:
            yield n