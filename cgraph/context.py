

class Context(object):
    def __init__(self):
        self.value = None
        self.ngradient = None
        self.sgradient = None
        self.cache = None


