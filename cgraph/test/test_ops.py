from pytest import approx

import cgraph.symbols as sym
from .test_helpers import checkf

def test_add():

    x = sym.Symbol('x')
    y = sym.Symbol('y')

    f = x + y
    checkf(f, {x:2, y:3}, value=5, ngrad={x: 1, y:1})

def test_sub():

    x = sym.Symbol('x')
    y = sym.Symbol('y')

    f = x - y
    checkf(f, {x:2, y:3}, value=-1, ngrad={x: 1, y:-1})

def test_mul():

    x = sym.Symbol('x')
    y = sym.Symbol('y')

    f = x * y
    checkf(f, {x:2, y:3}, value=6, ngrad={x: 3, y:2})