from pytest import approx
import math

import cgraph as cg


def test_add():

    x = cg.Symbol('x')
    y = cg.Symbol('y')

    f = x + y
    d = f.ndiff(x=3, y=4)
    assert d[x] == approx(1.)
    assert d[y] == approx(1.)

def test_sub():

    x = cg.Symbol('x')
    y = cg.Symbol('y')

    f = x - y
    d = f.ndiff(x=3, y=4)
    assert d[x] == approx(1.)
    assert d[y] == approx(-1.)

def test_mul():

    x = cg.Symbol('x')
    y = cg.Symbol('y')

    f = x * y
    d = f.ndiff(x=3, y=4)
    assert d[x] == approx(4)
    assert d[y] == approx(3)


def test_div():

    x = cg.Symbol('x')
    y = cg.Symbol('y')

    f = x / y
    d = f.ndiff(x=2, y=4)
    assert d[x] == approx(1/4)
    assert d[y] == approx(-2/4**2)


def test_neg():

    x = cg.Symbol('x')

    f = -x
    d = f.ndiff(x=2)
    assert d[x] == approx(-1)

def test_abs():

    x = cg.Symbol('x')

    f = abs(x)
    d = f.ndiff(x=-2)
    assert d[x] == approx(-2/abs(2))

def test_fun():

    x = cg.Symbol('x')
    y = cg.Symbol('y')
    z = cg.Symbol('z')

    f = (x * y + 3) / (z - 2)
    d = f.ndiff(x=3, y=4, z=4)

    assert d[x] == approx(2.)
    assert d[y] == approx(1.5)
    assert d[z] == approx(-3.75)

    k = x * 3 - cg.pi
    m = f / k

    d = m.ndiff(x=3, y=4, z=4)
    assert d[x] == approx(-0.3141868015256969)
    assert d[y] == approx(0.2560422844135512)
    assert d[z] == approx(-0.64010571103387)

    l = (-x + abs(f)) * m
    d = l.ndiff(x=3, y=4, z=4)
    assert d[x] == approx(-0.133629184797880)
    assert d[y] == approx(3.07250741296261)
    assert d[z] == approx(-7.68126853240654)
    

def test_circle_area():

    r = cg.Symbol('r')
    A = r * r * cg.pi

    d = A.ndiff(r=3)
    assert d[r] == approx(6*math.pi)
