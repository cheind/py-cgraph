"""CGraph - symbolic computation in Python library.

This library is the result of my efforts to understand symbolic computation of
functions factored as expression trees. In a few lines of code it shows how to
forward evaluate functions and how to perform numeric and symbolic derivatives
computations using backpropagation.

While this library is not complete (and will never be) it offers the interested
reader some insights on one way in which symbolic computation can be performed.

The code is accompanied by a series of notebooks that explain the fundamental
concepts. You can find these notebooks online at

    https://github.com/cheind/py-cgraph

Christoph Heindl, 2017
"""

from .cgraph import *