from collections import defaultdict
import numpy as np
import graph

class Node:
    def __init__(self, input_required=False):
        self.ins = []
        self.outs = []
        self.input_required = input_required
        
    def __str__(self):
        return self.name if self.name is not None else self.__class__.__name__    
        
    def __repr__(self):
        return self.__str__()

    def compute(self, inputs):
        pass

    @property
    def in_degree(self):
        return len(self.ins)

    @property
    def out_degree(self):
        return len(self.outs)
   
    def eval(self, **kwargs):   
        nodes = set(graph.parents(self))        
        order = graph.topological_sort(nodes)

        # Forward pass
        values = {}
        grads = defaultdict(dict)

        for n in order:
            if n.in_degree == 0 and n.input_required:
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

            if n == self:
                d = 1.
            else:                
                d = sum([v for v in grads[n].values()])

            diffs[n] = d
                
            for i in n.ins:
                grads[i][n] *= d

        return values[self], diffs 


    @staticmethod
    def add_edge(src, dst):
        if dst not in src.outs:
            src.outs.append(dst)

        if src not in dst.ins:
            dst.ins.append(src)

    @staticmethod
    def nary_function(klass, *args):   
        c = klass()
        for n in args:
            Node.add_edge(n, c)
        return c