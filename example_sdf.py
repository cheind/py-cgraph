
import numpy as np

import sdf
import cgraph as cg
import matplotlib.pyplot as plt
import matplotlib.colors as clr



x = cg.Symbol('x')
y = cg.Symbol('y')

# Support set ops?
# http://www.linuxtopia.org/online_books/programming_books/python_programming/python_ch16s03.html

"""
f = sdf.union(
    sdf.circle(x, y),
    sdf.circle(x, y, cx=0.6),
    sdf.circle(x, y, cx=-0.6, cy=-0.2, r=0.5))
"""

f = sdf.circle(x, y) | \
    sdf.circle(x, y, cx=0.8) | \
    sdf.circle(x, y, cx=-0.6, cy=-0.2, r=0.7)

ex, ey = np.mgrid[-2:2:100j, -2:2:100j]

def eval_sdf():
    r = np.empty(ex.shape)
    for ix in range(ex.shape[0]):
        for iy in range(ex.shape[1]):            
            r[ix, iy] = cg.value(f, {x: ex[ix, iy], y: ey[ix, iy]})
    return r





# http://stackoverflow.com/questions/25342072/computing-and-drawing-vector-fields
#r = eval_sdf()
#fig, ax = plt.subplots()
#norm = clr.Normalize(vmin = np.min(r), vmax = np.max(r), clip = False)
#ax.imshow(r, extent=[ex.min(), ex.max(), ey.min(), ey.max()], cmap='spectral', norm=norm)
#cont = ax.contour(ex, ey, r)
#ax.set_aspect('equal')
#plt.clabel(cont, inline=1, fontsize=10)
#plt.show()

X, Y = np.mgrid[-2:2:100j, -2:2:100j]

k = sdf.Function(f, [x, y])
r, g = k(X.reshape(-1, 1), Y.reshape(-1, 1), with_gradient=True)



fig, ax = plt.subplots()
img = ax.imshow(r.reshape(X.shape).transpose(), cmap='spectral', extent=[X.min(), X.max(), Y.min(), Y.max()], origin='lower')
cont = ax.contour(X, Y, r.reshape(X.shape))
ax.set_aspect('equal')
plt.clabel(cont, inline=1, fontsize=10)
plt.show()

