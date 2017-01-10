from cgraph.arithmetics import ArithmeticNode

class Symbol(ArithmeticNode):
    def __init__(self, name):
        super(Symbol, self).__init__()
        self.name = name

    def __str__(self):
        return self.name

    def __hash__(self):
        return hash(self.name)            
    
    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.name == other.name      
        else:
            return False
        
    def compute(self, inputs):
        pass