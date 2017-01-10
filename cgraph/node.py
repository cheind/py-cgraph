from collections import defaultdict

class Node:
    def __init__(self):
        self.ins = []
        self.outs = []
        
    def __str__(self):
        return self.name if self.name is not None else self.__class__.__name__    
        
    def __repr__(self):
        return self.__str__()

    def compute(self, inputs):
        pass

    @property
    def in_degree(self):
        return len(self.ins)

    def ins_with_multiplicity(self):
        d = defaultdict(lambda: 0)
        for n in self.ins:
            d[n] += 1
        return d

    @property
    def out_degree(self):
        return len(self.outs)

    @staticmethod
    def add_edge(src, dst):
        src.outs.append(dst)
        dst.ins.append(src)

    @staticmethod
    def nary_function(klass, *args):   
        c = klass()
        for n in args:
            Node.add_edge(n, c)
        return c