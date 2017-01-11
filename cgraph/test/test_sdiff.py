from pytest import approx
import math
import cgraph as cg

def test_add():

    x = cg.Symbol('x')
    y = cg.Symbol('y')

    f = (x + y)*x
    d = f.sdiff()
    
    
    print(d[x])
    print(d[y])

    #print(d[x].eval(x=2, y=3))
    
    