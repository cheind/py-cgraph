from pytest import approx

import cgraph as cg

def test_single_symbol():
    x = cg.Symbol('x')
    y = (x + 3) / 2.

    assert y.eval(x=1) == approx(2.)

def test_multiple_symbols(): 
    x = cg.Symbol('x')
    y = cg.Symbol('y')
    z = cg.Symbol('z')
    
    w = (x - 2) / (y * z)

    assert w.eval(x=4, y=2, z=4) == approx(1/4)
