from pytest import approx
import math

import cgraph as cg


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


def test_circle_area():

    r = cg.Symbol('r')
    A = r * r * cg.pi

    d = A.ndiff(r=3)
    assert d[r] == approx(6*math.pi)
