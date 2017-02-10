
## CGraph - symbolic computation in Python

This repository is the result of my efforts to understand symbolic computation of
functions factored as expression trees. With the right concepts at hands, a few lines of code are able compute numeric and symbolic gradients of arbitrary functions. Even expression simplification is easily accomplished. While this library is not complete (and will never be) it offers the interested reader some insights on how to perform symbolic computation.

The code is accompanied by a series of notebooks that explain fundamental concepts for developing your own library that performs symbolic computation:

**Foundations**
- Computational Graphs - Introduction [view][1]
- Computational Graphs - Symbolic Computation in Python [view][2]

**Applications**
- Function Optimization [view][3]
- Signed Distance Functions and Particle Physics [view][4]

All notebook sources can be found inside the [docs][docs] folder. There are certainly plenty of typos, grammatical errors and all kind of improvements possible. In case you have one for me, I'd be happy to see your pull requests or comments!

### Usage
The code for symbolic computation contained in [cgraph.py][cgraph.py] can be used as follows.

```python
import cgraph as cg

x = cg.Symbol('x')
y = cg.Symbol('y')
z = cg.Symbol('z')

f = (x * y + 3) / (z - 2)

# Evaluate function
cg.value(f, {x:2, y:3, z:3})        # 9.0

# Partial derivatives (numeric)
d = cg.numeric_gradient(f, {x:2, y:3, z:3})
d[x]                                # df/dx 3.0
d[z]                                # df/dz -9.0

# Partial derivatives (symbolic)
d = cg.symbolic_gradient(f)
cg.simplify(d[x])                   # df/dx (y*(1/(z - 2)))
cg.value(d[x], {x:2, y:3, z:3})     # df/dx 3.0

# Higher order derivatives
ddx = cg.symbolic_gradient(d[x])
cg.simplify(ddx[y])                 # ddf/dxdy (1/(z - 2))
```

### Installation
To install CGraph clone this repository and use `pip` to install

```
pip install -e <path/to/setup.py>
```

### License
If not otherwise stated all Material is licensed under BSD license.

[1]: https://cdn.rawgit.com/cheind/py-cgraph/master/docs/00_Computational_Graphs-Introduction.html
[2]: https://cdn.rawgit.com/cheind/py-cgraph/master/docs/01_Computational_Graphs-Symbolic_Computation.html
[3]: https://cdn.rawgit.com/cheind/py-cgraph/master/docs/02_Computational_Graphs-Function_Optimization.html
[4]: https://cdn.rawgit.com/cheind/py-cgraph/master/docs/03_Computational_Graphs-Signed_Distance_Functions_and_Particle_Physics.html

[cgraph.py]: cgraph/cgraph.py
[docs]: docs/