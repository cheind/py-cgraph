from pytest import approx

import cgraph as cg

def test_add():

    x = cg.Symbol('x')
    y = cg.Symbol('y')

    f = (x + y)*x
    d = f.sdiff() # symbolic diff with respect to all inputs

    print(d[x]) # gives ((1*(x + y)) + ((1*x)*1)), which simplifies to 2x + y
    assert d[x].eval(x=2, y=3) == approx(7)

    print(d[y]) # gives ((1*x)*1), which simplifies to x
    assert d[y].eval(x=2, y=3) == approx(2)
    
    