
import pytest
import math
import numpy as np

import cgraph as cg

def checkf(f, fargs, value=None, ngrad=None, with_sgrad=True):
    #__tracebackhide__ = True
   
    v = cg.value(f, fargs)
    if not all(np.isclose(value, v)):
        pytest.fail("""Function VALUE check failed
        f: {}
        expected value of {} - received {}""".format(f, value, v))

    if ngrad is not None:
        ng = cg.numeric_gradient(f, fargs)    
        for k in fargs.keys():
            if not all(np.isclose(ngrad[k], ng[k])):
                pytest.fail("""Function NUMERIC GRAD check failed
                f: {}, 
                df/d{}
                expected value of {} - received {}""".format(f, k, ngrad[k], ng[k]))

    if ngrad is not None and with_sgrad:
        ng = cg.numeric_gradient(f, fargs)
        sg = cg.symbolic_gradient(f) 
        for k in fargs.keys():
            if not all(np.isclose(ngrad[k], cg.value(sg[k], fargs))):
                pytest.fail("""Function SYMBOLIC GRAD check failed
                f: {}, 
                df/d{}: {},
                expected value of {} - received {}""".format(f, k, sg[k], ngrad[k], cg.value(sg[k], fargs)))

def complexity(f):
    count = 0
    for n in cg.postorder(f):
        if len(n.children) > 0:
            count += 1
    return count

def test_add():
    x = cg.Symbol('x')
    y = cg.Symbol('y')

    f = x + y
    checkf(f, {x:2, y:3}, value=5, ngrad={x: 1, y:1})
    checkf(f, {x:[2,3], y:[4,5]}, value=[6,8], ngrad={x:[1, 1], y:[1, 1]})

def test_sub():
    x = cg.Symbol('x')
    y = cg.Symbol('y')

    f = x - y
    checkf(f, {x:2, y:3}, value=-1, ngrad={x: 1, y:-1})
    checkf(f, {x:[2,3], y:[4,5]}, value=[-2,-2], ngrad={x:[1, 1], y:[-1, -1]})

def test_mul():
    x = cg.Symbol('x')
    y = cg.Symbol('y')

    f = x * y
    checkf(f, {x:2, y:3}, value=6, ngrad={x: 3, y:2})

def test_div():
    x = cg.Symbol('x')
    y = cg.Symbol('y')

    f = x / y
    checkf(f, {x:2, y:3}, value=2/3, ngrad={x: 1/3, y:-2/9})

def test_log():
    x = cg.Symbol('x')

    f = cg.sym_log(x)
    checkf(f, {x:2}, value=math.log(2), ngrad={x: 1/2})

def test_neg():
    x = cg.Symbol('x')

    f = -x
    checkf(f, {x:2}, value=-2, ngrad={x: -1})

def test_pow():
    x = cg.Symbol('x')
    y = cg.Symbol('y')

    f = x**y
    checkf(f, {x:2, y:3}, value=8, ngrad={x: 12, y:math.log(256)})

    d = cg.symbolic_gradient(f)
    checkf(d[x], {x:2, y:3}, value=12, ngrad={x: 12, y:4+math.log(4096)})                                      # ddf/dxdx and ddf/dxdy
    checkf(d[y], {x:2, y:3}, value=math.log(256), ngrad={x:4+math.log(4096), y:8 * math.log(2) * math.log(2)}) # ddf/dydx and ddf/dydy

    f = (x * 2 + y)**2
    checkf(f, {x:2, y:3}, value=7**2, ngrad={x: 2*7*2, y:2*7*1})

def test_exp():
    x = cg.Symbol('x')
    
    f = cg.sym_exp(x)
    checkf(f, {x:2}, value=math.exp(2), ngrad={x: math.exp(2)})
    checkf(f, {x:[1,0]}, value=[math.exp(1), math.exp(0)], ngrad={x: [math.exp(1), math.exp(0)]})

def test_sum():
    x = cg.Symbol('x')
    y = cg.Symbol('y')

    f = cg.sym_sum([x, y, x, y])
    checkf(f, {x:2, y:3}, value=10, ngrad={x: 2, y:2})

def test_sqrt():
    x = cg.Symbol('x')
    
    f = cg.sym_sqrt(x)
    checkf(f, {x:4}, value=2, ngrad={x: 0.25})

def test_reuse_of_expr():
    x = cg.Symbol('x')
    y = cg.Symbol('y')

    xy = x * y
    f = (xy + 1) * xy
    checkf(f, {x:2, y:3}, value=42, ngrad={x: 39, y:26})

def test_simplify_expr():
    x = cg.Symbol('x')
    y = cg.Symbol('y')

    f = x + 0
    s = cg.simplify(f)
    assert complexity(f) > complexity(s)
    assert cg.value(f, {x:2}) == cg.value(s, {x:2})

    f = x * 1
    s = cg.simplify(f)
    assert complexity(f) > complexity(s)
    assert cg.value(f, {x:2}) == cg.value(s, {x:2})

    f = x + x + x + x + x
    d = cg.symbolic_gradient(f)
    dx = cg.simplify(d[x])
    assert complexity(dx) == 0
    assert cg.is_const(dx, 5)


def test_complex_expr():
    
    x = cg.Symbol('x')
    y = cg.Symbol('y')
    z = cg.Symbol('z')

    f = (x * y + 3) / (z - 2)
    checkf(f, {x:3, y:4, z:4}, value=7.5, ngrad={x:2., y:1.5, z:-3.75})
    
    k = x * 3 - math.pi
    m = f / k
    checkf(
        m, {x:3, y:4, z:4}, 
        value=1.28021142206776, 
        ngrad={
            x:-0.314186801525697, 
            y:0.2560422844135512, 
            z:-0.64010571103387
        })

def test_function():
    x = cg.Symbol('x')
    y = cg.Symbol('y')

    e = (x * y + 3)
    f = cg.Function(e, [x, y])

    assert np.isclose(f(2, 1), 5)
    assert all(np.isclose(f([2,3], [1,2]), [5, 9]))

    v, g = f([2,3], [1,2], compute_gradient=True)
    assert all(np.isclose(v, [5, 9]))
    assert all(np.isclose(g[0], [1, 2]))
    assert all(np.isclose(g[1], [2, 3]))

