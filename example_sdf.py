
import numpy as np

import sdf
import cgraph as cg
import matplotlib.pyplot as plt
import matplotlib.colors as clr

x = cg.Symbol('x')
y = cg.Symbol('y')

c0 = sdf.circle(x, y)
c1 = sdf.circle(x, y, cx=0.6)
f = sdf.union(c0, c1)

ex, ey = np.mgrid[-2:2:100j, -2:2:100j]

def eval_sdf():
    r = np.empty(ex.shape)
    for ix in range(ex.shape[0]):
        for iy in range(ex.shape[1]):            
            r[ix, iy] = cg.value(f, {x: ex[ix, iy], y: ey[ix, iy]})
    return r

# http://stackoverflow.com/questions/25342072/computing-and-drawing-vector-fields
r = eval_sdf()
fig, ax = plt.subplots()
norm = clr.Normalize(vmin = np.min(r), vmax = np.max(r), clip = False)
ax.imshow(r, extent=[ex.min(), ex.max(), ey.min(), ey.max()], cmap='spectral', norm=norm)
plt.show()



