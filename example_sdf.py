
import numpy as np

import sdf
import cgraph as cg
import matplotlib.pyplot as plt
from matplotlib import animation

sx = cg.Symbol('x')
sy = cg.Symbol('y')

#f = sdf.circle(sx, sy) | \
#    sdf.circle(sx, sy, cx=0.8) | \
#    sdf.circle(sx, sy, cx=-0.6, cy=-0.2, r=0.7)
f = sdf.line(sx, sy, n=[0, 1], d=-1.8) | sdf.line(sx, sy, n=[1, 1], d=-1.8) | sdf.line(sx, sy, n=[-1, 1], d=-1.8) | sdf.circle(sx, sy, c=[0, -0.8], r=0.5)
k = sdf.Function(f, [sx, sy])


class Particles:

    def __init__(self, n):
        self.n = n
        self.x = np.zeros((n, 2))
        self.v = np.zeros((n, 2))
        self.m = np.ones(n)
        self.f = np.zeros((n, 2))
        self.dx = np.zeros((n, 2))
        self.dv = np.zeros((n, 2))

        self.actors = [plt.Circle((0,0), radius=0.08) for i in range(n)]

    def update_forces(self, t):
        self.f.fill(0.)
        self.f[:, 1] = self.m * -1. 
        return self.f

    def update_dynamics(self, t):
        self.dx[:] = self.v
        self.dv[:] = self.update_forces(t) / self.m[:, np.newaxis]
        return self.dx, self.dv

    def update_state(self, sx, sv):
        self.x[:] = sx
        self.v[:] = sv

        for i, a in enumerate(self.actors):
            a.center = self.x[i,:]

    @staticmethod
    def integrate_euler(sx, sv, dx, dv, h):
        return sx + dx * h, sv + dv * h


n=20
p = Particles(n)
p.x = np.random.multivariate_normal([0, 1], [[0.1, 0],[0, 0.1]], n)
p.v = np.random.multivariate_normal([0, 0], [[0.1, 0],[0, 0.1]], n)


X, Y = np.mgrid[-2:2:100j, -2:2:100j]
r, g = k(X.reshape(-1, 1), Y.reshape(-1, 1), with_gradient=True)
shape = X.shape
v = r.reshape(shape)
dx = g[:,0].reshape(shape)
dy = g[:,1].reshape(shape)

fig, ax = plt.subplots()
ax.set_xlim((-2, 2))
ax.set_ylim((-2, 2))
ax.set_aspect('equal')
cont = ax.contour(X, Y, v, levels=[0])
skip = (slice(None, None, 5), slice(None, None, 5))
ax.quiver(X[skip], Y[skip], dx[skip], dy[skip])

def init():    
    for c in p.actors:
        ax.add_patch(c)

    return p.actors
    
h = 0.02
def animate(i):

    dbefore = k(p.x[:,0], p.x[:,1])
    dx, dv = p.update_dynamics(0)
    sx, sv = Particles.integrate_euler(p.x, p.v, dx, dv, h)

    dafter, g = k(sx[:,0], sx[:,1], with_gradient=True)
    n = g / np.linalg.norm(g, axis=1)[:,np.newaxis]

    cids = np.where(dafter <= 0)[0]

    if len(cids) > 0:
        cr = 0.7
        cf = 0.2
        # Update pos
        sx[cids, :] = sx[cids, :] - (1+cr) * dafter[cids, np.newaxis] * n[cids, :]
        # Update velocity
        vn = np.sum(sv[cids, :] * n[cids, :], axis=1)[:,np.newaxis] * n[cids, :]
        vt = sv[cids, :] - vn
        sv[cids, :] = -cr * vn + (1 - cf)*vt
    
    p.update_state(sx, sv)
    return p.actors

anim = animation.FuncAnimation(fig, animate,  
                               frames=2000, 
                               interval=10,
                               init_func=init,
                               repeat=False,
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
