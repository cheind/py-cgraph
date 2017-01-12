from pytest import approx
import math

from .test_helpers import checkf

import cgraph.symbols as sym
import cgraph.ops.sign
import cgraph.ops.logarithm
import cgraph.ops.absolute
import cgraph.ops.exponential

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

def test_div():
    x = sym.Symbol('x')
    y = sym.Symbol('y')

    f = x / y
    checkf(f, {x:2, y:3}, value=2/3, ngrad={x: 1/3, y:-2/9})

def test_abs():
    x = sym.Symbol('x')
    f = abs(x)
    checkf(f, {x:-2}, value=2, ngrad={x: -1})

    f = cgraph.ops.absolute.sym_abs(x)
    checkf(f, {x:-2}, value=2, ngrad={x: -1})


def test_signum():
    x = sym.Symbol('x')
    f = cgraph.ops.sign.sym_sgn(x)
    checkf(f, {x:-2}, value=-1, ngrad={x: 0})

def test_log():
    x = sym.Symbol('x')
    f = cgraph.ops.logarithm.sym_log(x)
    checkf(f, {x:2}, value=math.log(2), ngrad={x: 1/2})


def test_pow():
    x = sym.Symbol('x')
    y = sym.Symbol('y')

    f = x**y
    checkf(f, {x:2, y:3}, value=8, ngrad={x: 12, y:math.log(256)})    

    d = f.sdiff()
    checkf(d[x], {x:2, y:3}, value=12, ngrad={x: 12, y:4+math.log(4096)})                                      # ddf/dxdx and ddf/dxdy
    checkf(d[y], {x:2, y:3}, value=math.log(256), ngrad={x:4+math.log(4096), y:8 * math.log(2) * math.log(2)}) # ddf/dydx and ddf/dydy

def test_exp():

    x = sym.Symbol('x')
    f = cgraph.ops.exponential.sym_exp(x) / x
    checkf(f, {x:2}, value=math.exp(2)/2, ngrad={x: math.exp(2)/4})