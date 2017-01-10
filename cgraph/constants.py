from cgraph.arithmetics import ArithmeticNode

class Constant(ArithmeticNode):
    def __init__(self, value):
        super(Constant, self).__init__()
        self.value = value

    def __str__(self):
        return str(self.value)

    """ 
    Constants are wrapped on the fly, so many such
    objects might be generated. We cannot hash those
    constants by value, as we are not obtaining a global graph
    structure. Therefore, we would just lose information.
    def __hash__(self):
        return hash(self.value)            
    
    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.value == other.value      
        else:
            return False
    """

    def compute(self, inputs):
        return self.value, [0.]