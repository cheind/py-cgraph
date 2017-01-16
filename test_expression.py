
import pytest
from pytest import approx

import expression as exp

def checkf(f, fargs, value=None, ngrad=None):
    __tracebackhide__ = True
   
    v = exp.value(f, fargs)
    if value != approx(v):
        pytest.fail("""Function VALUE check failed
        f: {}
        expected value of {} - received {}""".format(f, value, v))

    ng = exp.numeric_gradient(f, fargs)
    sg = exp.symbolic_gradient(f)

    for k in fargs.keys():
        if ngrad[k] != approx(ng[k]):
            pytest.fail("""Function NUMERIC GRAD check failed
            f: {}, 
            df/d{}
            expected value of {} - received {}""".format(f, k, ngrad[k], ng[k]))

        if ngrad[k] != approx(exp.value(sg[k], fargs)):
            pytest.fail("""Function SYMBOLIC GRAD check failed
            f: {}, 
            df/d{}: {},
            expected value of {} - received {}""".format(f, k, sg[k], ngrad[k], exp.value(sg[k], fargs)))


def test_add():
    x = exp.Symbol('x')
    y = exp.Symbol('y')

    f = x + y
    checkf(f, {x:2, y:3}, value=5, ngrad={x: 1, y:1})

def test_sub():
    x = exp.Symbol('x')
    y = exp.Symbol('y')

    f = x - y
    checkf(f, {x:2, y:3}, value=-1, ngrad={x: 1, y:-1})

def test_mul():
    x = exp.Symbol('x')
    y = exp.Symbol('y')

    f = x * y
    checkf(f, {x:2, y:3}, value=6, ngrad={x: 3, y:2})

def test_sqr():
    x = exp.Symbol('x')
    y = exp.Symbol('y')

    f = exp.sym_sqr(x * 2 + y)
    checkf(f, {x:2, y:3}, value=7**2, ngrad={x: 2*7*2, y:2*7*1})

def test_div():
    x = exp.Symbol('x')
    y = exp.Symbol('y')

    f = x / y
    checkf(f, {x:2, y:3}, value=2/3, ngrad={x: 1/3, y:-2/9})

def test_reuse_of_expr():
    x = exp.Symbol('x')
    y = exp.Symbol('y')

    xy = x * y
    f = (xy + 1) * xy
    checkf(f, {x:2, y:3}, value=42, ngrad={x: 39, y:26})