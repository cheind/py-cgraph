
import numpy as np

import sdf
import cgraph as cg
import matplotlib.pyplot as plt
import matplotlib.colors as clr

x = cg.Symbol('x')
y = cg.Symbol('y')

f = sdf.circle(x, y) | \
    sdf.circle(x, y, cx=0.8) | \
    sdf.circle(x, y, cx=-0.6, cy=-0.2, r=0.7)

X, Y = np.mgrid[-2:2:100j, -2:2:100j]

k = sdf.Function(f, [x, y])
r, g = k(X.reshape(-1, 1), Y.reshape(-1, 1), with_gradient=True)

fig, ax = plt.subplots()

shape = X.shape
v = r.reshape(shape)
dx = g[:,0].reshape(shape)
dy = g[:,1].reshape(shape)

# quiver
skip = (slice(None, None, 5), slice(None, None, 5))
ax.quiver(X[skip], Y[skip], dx[skip], dy[skip])
# or colored
# ax.quiver(X[skip], Y[skip], dx[skip], dy[skip], v[skip])

img = ax.imshow(v.transpose(), cmap='spectral', extent=[X.min(), X.max(), Y.min(), Y.max()], origin='lower')
cont = ax.contour(X, Y, v)
ax.set_aspect('equal')
plt.clabel(cont, inline=1, fontsize=10)
plt.show()

