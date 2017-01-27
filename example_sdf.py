
import numpy as np

import sdf
import cgraph as cg
import matplotlib.pyplot as plt
from matplotlib import animation

sx = cg.Symbol('x')
sy = cg.Symbol('y')

f = sdf.circle(sx, sy) | \
    sdf.circle(sx, sy, cx=0.8) | \
    sdf.circle(sx, sy, cx=-0.6, cy=-0.2, r=0.7)
k = sdf.Function(f, [sx, sy])


s = np.array([
    [0.0, 2.0], # pos
    [0.1, 0.0], # vel
])
n = int(s.shape[0] / 2)
m = np.ones(n)

def forces(s, t):
    f = np.zeros((n, 2))
    f[:, 1] = m * -1.    
    return f

def dynamics(s, t):
    d = np.empty(s.shape)
    d[:n,:] = s[n:,:] # dx/dt = v
    d[n:,:] = forces(s, t) / m
    return d


def update(s, d, h):
    return s + d * h

X, Y = np.mgrid[-2:2:100j, -2:2:100j]
r, g = k(X.reshape(-1, 1), Y.reshape(-1, 1), with_gradient=True)
shape = X.shape
v = r.reshape(shape)
dx = g[:,0].reshape(shape)
dy = g[:,1].reshape(shape)

c = plt.Circle((0, 0), 0.2)
fig, ax = plt.subplots()
ax.set_xlim((-2, 2))
ax.set_ylim((-2, 2))
ax.set_aspect('equal')
ax.add_patch(c)
cont = ax.contour(X, Y, v)

def init():
    c.center = s[0,:]
    return c,

h = 0.02
def animate(i):
    d = dynamics(s, 0)
    snew = update(s, d, h)
    s[:] = snew
    c.center = s[0,:]
    return c,    

anim = animation.FuncAnimation(fig, animate, 
                               init_func=init, 
                               frames=360, 
                               interval=20,
                               blit=True)
plt.show()

"""
X, Y = np.mgrid[-2:2:100j, -2:2:100j]
r, g = k(X.reshape(-1, 1), Y.reshape(-1, 1), with_gradient=True)
fig, ax = plt.subplots()

shape = X.shape
v = r.reshape(shape)
dx = g[:,0].reshape(shape)
dy = g[:,1].reshape(shape)

#interpolation
#http://stackoverflow.com/questions/12729228/simple-efficient-bilinear-interpolation-of-images-in-numpy-and-python

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
"""
