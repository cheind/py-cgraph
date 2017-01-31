
import numpy as np
import time

import sdf
import cgraph as cg
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.collections import PatchCollection

f = sdf.Line(normal=[0, 1], d=-1.8) | sdf.Line(normal=[1, 1], d=-1.8) | sdf.Line(normal=[-1, 1], d=-1.8)
#f = f | sdf.Circle(center=[0, -0.8], radius=0.5) - sdf.Circle(center=[0, -0.5], radius=0.5)
f = f | (sdf.Circle(center=[0, -0.8], radius=0.5) & sdf.Line(normal=[0.1, 1], d=-0.5))

def gravity(p, t):
    return p['m'][:, np.newaxis] * np.array([0, -1]) 


def explicit_euler(x, v, dx, dv, t, dt):
    xnew = np.empty(x.shape)
    vnew = np.empty(v.shape)
    
    xnew = x + dx * dt
    vnew = v + dv * dt

    return xnew, vnew

def timeit(f):
    def wrap(*args):
        time1 = time.time()
        ret = f(*args)
        time2 = time.time()
        print('Took {}'.format(time2-time1))
        return ret
    return wrap

class ParticleSimulator:

    def __init__(self, f, particle_creator, timestep=1/30):
        self.p = None        
        self.dt = timestep
        
        self.sdf = f
        self.particle_creator = particle_creator
        self.force_generators = []
        self.integrator = explicit_euler       
        
    def reset(self):
        self.p = self.particle_creator()
        self.p['f'] = np.zeros((self.p['n'], 2))

        self.t = 0
        self.current_wall_time = time.time()
        self.tacc = 0.      

    def forces(self):
        facc = self.p['f']
        
        facc.fill(0.)
        for fg in self.force_generators:
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
        new_wall_time = time.time()
        frame_time = new_wall_time - self.current_wall_time
        self.current_wall_time = new_wall_time
        self.tacc += frame_time

        while self.tacc >= self.dt:
            self.advance()
            self.tacc -= self.dt
            self.t += self.dt
    
    @timeit
    def advance(self):
        xcur = self.p['x']
        vcur = self.p['v']

        dx, dv = self.dynamics()
        xnew, vnew = self.integrator(xcur, vcur, dx, dv, self.t, self.dt)
        
        d, g = self.sdf(xnew[:, 0], xnew[:, 1], compute_gradient=True)
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

def create_particles(n=1000):
    p = {}
    p['n'] = n
    p['x'] = np.random.multivariate_normal([0, 1], [[0.05, 0],[0, 0.05]], n)
    p['v'] = np.random.multivariate_normal([0, 0], [[0.1, 0],[0, 0.1]], n)
    p['m'] = np.random.uniform(1, 10, size=n)
    p['r'] = p['m'] * 0.01
    p['cr'] = np.full(n, 0.6)
    p['cf'] = np.full(n, 0.3)
    return p

def plot_background(fig, ax, f, bounds=[(-2,2), (-2,2)], show_quiver=True, show_isolines='all'):
    
    ax.set_xlim(bounds[0])
    ax.set_ylim(bounds[1])
    ax.set_aspect('equal')
    ax.tick_params(
        axis='both',
        which='both',
        bottom='off',
        top='off',
        left='off',
        right='off',
        labelbottom='off',
        labeltop='off',
        labelleft='off',
        labelright='off',
    )

    if show_quiver or show_isolines:
        x, y, d, g = sdf.grid_eval(f, bounds=bounds)
        
        if show_isolines == 'all':
            cont = ax.contour(x, y, d)
        elif show_isolines == 'zero':
            cont = ax.contour(x, y, d, levels=[0])       

        if show_quiver:
            dx = g[:,:,0]
            dy = g[:,:,1]

            skip = (slice(None, None, 5), slice(None, None, 5))
            ax.quiver(x[skip], y[skip], dx[skip], dy[skip], d[skip])


def create_animation(fig, ax, ps, bounds=[(-2,2), (-2,2)], frames=500, timestep=1/30, repeat=True):
    patch = None

    def init_anim():
        global patch

        ps.reset()
        actors = [plt.Circle((0,0), radius=ps.p['r'][i]) for i in range(ps.p['n'])]        
        patch = ax.add_artist(PatchCollection(actors, offset_position='data', alpha=0.6, zorder=10))
        patch.set_array(np.random.rand(len(actors)))
        return patch,

    def update_anim(i):
        global patch

        ps.update()        
        patch.set_offsets(ps.p['x'])
        return patch,

    anim = animation.FuncAnimation(
        fig, 
        update_anim,  
        init_func=init_anim,
        interval=timestep * 1000,
        frames=frames,
        repeat=repeat,
        blit=True)

    return anim

ps = ParticleSimulator(f, create_particles, timestep=1/60)
ps.force_generators += [gravity]

fig, ax = plt.subplots()
plot_background(fig, ax, f, show_quiver=True, show_isolines='zero')
anim = create_animation(fig, ax, ps)
plt.show()