
import numpy as np
import time

import sdf
import cgraph as cg
import matplotlib.pyplot as plt
from matplotlib import animation

f = sdf.Line(normal=[0, 1], d=-1.8) | sdf.Line(normal=[1, 1], d=-1.8) | sdf.Line(normal=[-1, 1], d=-1.8)
#f = f | sdf.Circle(center=[0, -0.8], radius=0.5) - sdf.Circle(center=[0, -0.5], radius=0.5)
f = f | (sdf.Circle(center=[0, -0.8], radius=0.5) & sdf.Line(normal=[0.1, 1], d=-0.5))

#f = sdf.line(sx, sy, n=[0, 1], d=-1.8) | sdf.line(sx, sy, n=[1, 1], d=-1.8) | sdf.line(sx, sy, n=[-1, 1], d=-1.8) | (sdf.subtract(sdf.circle(sx, sy, c=[0, -0.8], r=0.5), sdf.circle(sx, sy, c=[0, -0.5], r=0.5)))


def create_particles(n=100):
    p = {}
    p['n'] = n
    p['x'] = np.zeros((n, 2))
    p['v'] = np.zeros((n, 2))
    p['m'] = np.ones(n)
    p['r'] = np.ones(n)
    p['cr'] = np.full(n, 0.6)
    p['cf'] = np.full(n, 0.3)
    return p


n = 100
particles = create_particles(n=n)
particles['x'] = np.random.multivariate_normal([0, 1], [[0.05, 0],[0, 0.05]], n)
particles['v'] = np.random.multivariate_normal([0, 0], [[0.1, 0],[0, 0.1]], n)
particles['m'] = np.random.uniform(1, 10, size=n)
particles['r'] = particles['m'] * 0.01


def gravity(p, t):
    return p['m'][:, np.newaxis] * np.array([0, -1]) 


def explicit_euler(x, v, dx, dv, t, dt):
    xnew = np.empty(x.shape)
    vnew = np.empty(v.shape)
    
    xnew = x + dx * dt
    vnew = v + dv * dt

    return xnew, vnew


class ParticleSimulation:

    def __init__(self, particles, forcegens, f, timestep=1/30, integrator=explicit_euler):
        self.f = f
        self.p = particles
        self.p['f'] = np.zeros((self.p['n'], 2))

        self.forcegens = forcegens
        self.dt = timestep
        self.t = 0
        self.integrator = integrator
        self.current_wall_time = None

    def forces(self):
        facc = self.p['f']
        
        facc.fill(0.)
        for fg in self.forcegens:
            k = fg(self.p, self.t)
            facc += k

        return facc

    def dynamics(self):
        dx = np.empty((self.p['n'], 2))
        dv = np.empty((self.p['n'], 2))

        dx[:] = self.p['v']
        dv[:] = self.forces() / self.p['m'][:, np.newaxis]

        return dx, dv

    def update(self):
        if self.current_wall_time is None:
            self.current_wall_time = time.time()
            self.tacc = 0.      

        new_wall_time = time.time()
        frame_time = new_wall_time - self.current_wall_time
        self.current_wall_time = new_wall_time
        self.tacc += frame_time

        while self.tacc >= self.dt:
            self.advance()
            self.tacc -= self.dt
            self.t += self.dt
        
    def advance(self):
        xcur = self.p['x']
        vcur = self.p['v']

        dx, dv = self.dynamics()
        xnew, vnew = self.integrator(xcur, vcur, dx, dv, self.t, self.dt)
        
        d, g = self.f(xnew[:, 0], xnew[:, 1], compute_gradient=True)
        d -= self.p['r'] # Correct for radius of particles

        cids = np.where(d <= 0)[0]
        if len(cids) > 0:
            # Collision response for items in collision     
            g = g[cids]
            n = g / np.linalg.norm(g, axis=1)[:, np.newaxis]

            x = xnew[cids]
            v = vnew[cids]
            cr = self.p['cr'][cids, np.newaxis]
            cf = self.p['cf'][cids, np.newaxis]

            vn = np.sum(v * n, axis=1)[:, np.newaxis] * n
            vt = v - vn

            xnew[cids] = x - (1 + cr) * d[cids, np.newaxis] * n
            vnew[cids] = -cr * vn + (1 - cf) * vt
        
        self.p['x'][:] = xnew
        self.p['v'][:] = vnew


fig, ax = plt.subplots()
ax.set_xlim((-2, 2))
ax.set_ylim((-2, 2))
ax.set_aspect('equal')
ax.axis('off')

X, Y = np.mgrid[-2:2:100j, -2:2:100j]
r, g = f(X.reshape(-1), Y.reshape(-1), compute_gradient=True)

shape = X.shape
v = r.reshape(shape)
dx = g[:,0].reshape(shape)
dy = g[:,1].reshape(shape)

cont = ax.contour(X, Y, v, levels=[0])
#cont = ax.contour(X, Y, v)
skip = (slice(None, None, 5), slice(None, None, 5))
ax.quiver(X[skip], Y[skip], dx[skip], dy[skip], v)

actors = [plt.Circle((0,0), radius=particles['r'][i]) for i in range(n)]
for c in actors:
    ax.add_patch(c)

ps = ParticleSimulation(particles, [gravity], f, timestep=1/60)

def animate(i):
    ps.update()
    for i, a in enumerate(actors):
        a.center = particles['x'][i, :]
    return actors

anim = animation.FuncAnimation(fig, animate,  
                               frames=1000, 
                               interval=1000/30,
                               repeat=False,
                               blit=True)
plt.show()
