from pytest import approx

from .test_helpers import checkf

import cgraph.symbols as sym
import cgraph.ops.sign

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


def test_abs():
    x = sym.Symbol('x')
    f = abs(x)
    checkf(f, {x:-2}, value=2, ngrad={x: -1})


def test_signum():
    x = sym.Symbol('x')
    f = cgraph.ops.sign.sgn(x)
    checkf(f, {x:-2}, value=-1, ngrad={x: 0})

