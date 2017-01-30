
import numpy as np
import time

import sdf
import cgraph as cg
import matplotlib.pyplot as plt
from matplotlib import animation

sx = cg.Symbol('x')
sy = cg.Symbol('y')

f = sdf.line(sx, sy, n=[0, 1], d=-1.8) | sdf.line(sx, sy, n=[1, 1], d=-1.8) | sdf.line(sx, sy, n=[-1, 1], d=-1.8) | (sdf.subtract(sdf.circle(sx, sy, c=[0, -0.8], r=0.5), sdf.circle(sx, sy, c=[0, -0.5], r=0.5)))
F = cg.Function(f, [sx, sy])

n = 100
stype = np.dtype([('x', float, 2), ('v', float, 2)])
p = np.zeros(n, dtype=[
    ('s', stype), 
    ('m', float),
    ('f', float, 2),
    ('r', float),
    ('cr', float),
    ('cf', float)
])

p['s']['x'] = np.random.multivariate_normal([0, 1], [[0.05, 0],[0, 0.05]], n)
p['s']['v'] = np.random.multivariate_normal([0, 0], [[0.1, 0],[0, 0.1]], n)
p['m'] = np.random.uniform(1, 10, size=n)
p['cr'] = 0.6
p['cf'] = 0.4
p['r'] = p['m'] * 0.01


def gravity(p, t):
    p['f'][:, 1] += p['m'] * -1


def explicit_euler(s, sd, t, dt):
    snew = np.empty(len(s), dtype=stype)
    snew['x'] = s['x'] + sd['x']*dt
    snew['v'] = s['v'] + sd['v']*dt
    return snew


class ParticleSimulation:

    def __init__(self, particles, forcegens, f, timestep=1/30, integrator=explicit_euler):
        self.f = f
        self.p = particles
        self.forcegens = forcegens
        self.dt = timestep
        self.t = 0
        self.int = integrator
        self.current_wall_time = None

    def forces(self, t):
        self.p['f'].fill(0.)
        for fg in self.forcegens:
            fg(p, t)
        return self.p['f']

    def dynamics(self, t):
        s = self.p['s']
        sd = np.empty(len(self.p), dtype=stype)        
        sd['x'] = s['v']
        sd['v'] = self.forces(t) / self.p['m'][:, np.newaxis]
        return sd

    def update(self):
        if self.current_wall_time is None:
            self.current_wall_time = time.time()
            self.tacc = 0.      

        new_wall_time = time.time()
        frame_time = new_wall_time - self.current_wall_time
        self.current_wall_time = new_wall_time
        self.tacc += frame_time

        while self.tacc >= self.dt:
            self.step()
            self.tacc -= self.dt
            self.t += self.dt
        
    def step(self):
        s = p['s']
        sd = self.dynamics(self.t)
        snew = self.int(s, sd, self.t, self.dt)

        x = s['x']
        xnew = snew['x']
        
        dbefore = self.f(x[:, 0], x[:, 1])
        dafter, g = self.f(xnew[:, 0], xnew[:, 1], compute_gradient=True)
        

        dafter -= p['r'] # Correct for radius

        cids = np.where(dafter <= 0)[0]
        if len(cids) > 0:
            # Collision response for items in collision     
            g = g[cids]
            n = g / np.linalg.norm(g, axis=1)[:, np.newaxis]

            xnew = snew['x'][cids]
            vnew = snew['v'][cids]
            cr = p['cr'][cids, np.newaxis]
            cf = p['cf'][cids, np.newaxis]

            xnew -= (1 + cr) * dafter[cids, np.newaxis] * n
            
            vn = np.sum(vnew * n, axis=1)[:, np.newaxis] * n
            vt = vnew - vn

            snew['x'][cids] = xnew
            snew['v'][cids] = -cr * vn + (1 - cf) * vt
        
        p['s'] = snew


fig, ax = plt.subplots()
ax.set_xlim((-2, 2))
ax.set_ylim((-2, 2))
ax.set_aspect('equal')
ax.axis('off')


X, Y = np.mgrid[-2:2:100j, -2:2:100j]
r, g = F(X.reshape(-1), Y.reshape(-1), compute_gradient=True)

#r, g = k(X.reshape(-1, 1), Y.reshape(-1, 1), with_gradient=True)
shape = X.shape
v = r.reshape(shape)
dx = g[:,0].reshape(shape)
dy = g[:,1].reshape(shape)

cont = ax.contour(X, Y, v, levels=[0])
#cont = ax.contour(X, Y, v)
skip = (slice(None, None, 5), slice(None, None, 5))
ax.quiver(X[skip], Y[skip], dx[skip], dy[skip], v)

actors = [plt.Circle((0,0), radius=p['r'][i]) for i in range(n)]
for c in actors:
    ax.add_patch(c)

ps = ParticleSimulation(p, [gravity], F, timestep=1/60)

def animate(i):
    ps.update()
    for i, a in enumerate(actors):
        a.center = p['s']['x'][i, :]
    return actors

anim = animation.FuncAnimation(fig, animate,  
                               frames=1000, 
                               interval=1000/30,
                               repeat=False,
                               blit=True)
plt.show()
