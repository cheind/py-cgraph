
import pytest
import math

from test_cgraph import checkf

import cgraph as cg
import sdf


def test_sdf_circle():
    x = cg.Symbol('x')
    y = cg.Symbol('y')

    f = sdf.circle(x, y, cx=1., cy=1., r=1)
    checkf(f, {x:2, y:1}, value=0., ngrad={x:1, y:0})
    # https://www.wolframalpha.com/input/?i=d%2Fdy+sqrt((x-1)%5E2%2B(y-1)%5E2)+-+1,+at+x%3D2,+y%3D1
    checkf(f, {x:0.9, y:0.7}, value=-0.683772, ngrad={x:-0.316228, y:-0.948683})
    checkf(f, {x:5, y:4}, value=4, ngrad={x:0.8, y:0.6})
