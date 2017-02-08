"""A 2D particle physics simulation using signed distance functions for collision test and response."""

import numpy as np
import time

import cgraph as cg
import cgraph.sdf as sdf

def timeit(f):
    def wrap(*args):
        time1 = time.time()
        ret = f(*args)
        time2 = time.time()
        print('Took {}'.format(time2-time1))
        return ret
    return wrap

def explicit_euler(x, v, dx, dv, t, dt):
    """Evolve in ODE in time by performing a single explicit euler step."""
    
    return x + dx * dt, v + dv * dt

class ParticleSimulator:
    """Particle based physics in 2D.
    
    ParticleSimulator takes a callable `particle_creator` and a signed 
    distance function `f` that represents the static environment particles
    live in and interact with.

    The particle creator function shall return any number of particles
    when called. The way the particles shall be organized is in a dictionary
    having at least the following properties set

    {
        'n': int, the number of particles
        'x': array of nx2 float, initial position of particles
        'v': array of nx2 float, initial velocity of particles
        'm': array of n float, particle masses
        'r': array of n float, particle radii
        'cr': array of n float, coefficient of restitution
        'cf': array of n float, coefficient of friction
    }

    By default no forces during simulation apply. Use `force_generators` property
    to add new forces to the array of forces. Each force generator is a callable
    that takes two arguments (particle state, time) and shell return a nx2 float
    array of forces for each particle.
    
    """

    def __init__(self, f, particle_creator, timestep=1/30):

        self.p = None        
        self.dt = timestep
        
        self.sdf = f
        self.particle_creator = particle_creator
        self.force_generators = []
        self.integrator = explicit_euler       
        
    def reset(self):
        """Reset simulation."""

        self.p = self.particle_creator()
        self.p['f'] = np.zeros((self.p['n'], 2))

        self.t = 0        
        self.tacc = 0.
        self.current_wall_time = time.time()      

    def forces(self):
        """Compute net force for each particle."""

        facc = self.p['f']
        
        facc.fill(0.)
        for fg in self.force_generators:
            k = fg(self.p, self.t)
            facc += k

        return facc

    def dynamics(self):
        """Compute state dynamics using Newton's second law."""

        dx = np.empty((self.p['n'], 2))
        dv = np.empty((self.p['n'], 2))

        dx[:] = self.p['v']
        dv[:] = self.forces() / self.p['m'][:, np.newaxis]

        return dx, dv

    #@timeit
    def update(self):
        """Update the simulation.
        
        The ParticleSimulator uses a fixed timestepping scheme.
        In order to determine the number of simulation rounds
        required, ParticleSimulator measures the elapsed time
        since last invocation. When the accumulated time is
        greater than the timestep it performs one or more steps.
        """

        new_wall_time = time.time()
        frame_time = new_wall_time - self.current_wall_time
        self.current_wall_time = new_wall_time
        self.tacc += frame_time

        while self.tacc >= self.dt:
            self.advance()
            self.tacc -= self.dt
            self.t += self.dt
    
    
    def advance(self):
        """Advance time by one timestep."""

        # Compute dynamics and state at t + dt
        xcur = self.p['x']
        vcur = self.p['v']

        dx, dv = self.dynamics()
        xnew, vnew = self.integrator(xcur, vcur, dx, dv, self.t, self.dt)
        
        # For collision test we query the signed distance function at the 
        # positions t + dt
        d, g = self.sdf(xnew[:, 0], xnew[:, 1], compute_gradient=True)

        # Since our particles are little circles we need to account for
        # their radius
        d -= self.p['r'] 

        # The particles in collision are those whose signed distance is 
        # equal to or less than 0
        cids = np.where(d <= 0)[0]
        if len(cids) > 0:
            # Collision response for affected particles
            # We use the gradient at the particles positon to determine
            # the direction in which the particle is moved to be pushed
            # outside of the object.

            g = g[cids]
            n = g / np.linalg.norm(g, axis=1)[:, np.newaxis]

            # Shortcuts
            x = xnew[cids]
            v = vnew[cids]
            cr = self.p['cr'][cids, np.newaxis]
            cf = self.p['cf'][cids, np.newaxis]

            # Normal and tangential components of particle velocity w.r.t the collision normal
            vn = np.sum(v * n, axis=1)[:, np.newaxis] * n
            vt = v - vn

            # Update position and velocity. See "Foundations of physically based modelling" page 55
            xnew[cids] = x - (1 + cr) * d[cids, np.newaxis] * n
            vnew[cids] = -cr * vn + (1 - cf) * vt
        
        self.p['x'][:] = xnew
        self.p['v'][:] = vnew


import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.collections import PatchCollection

def setup_axes(ax, bounds=[(-2,2), (-2,2)]):
    """Set default matplotlib axis properties."""
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

def plot_sdf(fig, ax, f, bounds=[(-2,2), (-2,2)], show_quiver=True, show_isolines='all'):
    """Plot a signed distance function.

    This function plots the contours of the signed distance function and additionally 
    is able to visualize gradients.
    """

    setup_axes(ax, bounds)

    ret = {}    

    if show_quiver or show_isolines:
        x, y, d, g = sdf.grid_eval(f, bounds=bounds)
        
        if show_isolines == 'all':
            ret['contour'] = ax.contour(x, y, d)
        elif show_isolines == 'zero':
            ret['contour'] = ax.contour(x, y, d, levels=[0])       

        if show_quiver:
            dx = g[:,:,0]
            dy = g[:,:,1]

            skip = (slice(None, None, 5), slice(None, None, 5))
            ret['quiver'] = ax.quiver(x[skip], y[skip], dx[skip], dy[skip], d[skip])
            
    return ret

def create_animation(fig, ax, ps, bounds=[(-2,2), (-2,2)], frames=500, timestep=1/30, repeat=True):
    """Create a matplotlib animation involving a particle simulation."""

    patches = []

    def init_anim():
        setup_axes(ax, bounds)      
        ps.reset()
        return []

    def update_anim(i):
        if i == 0:
            # We initialize the circles in here. It seems like matplotlib keeps a static image
            # of all circles at (0,0) when calling the same method inside init_anim. Also, we need a
            # new circle collection when an animation repeats, because radii of circles might have 
            # changed during ps.reset()            
            if len(patches) > 0:
                patches[0].remove()
                patches.pop()

            actors = [plt.Circle((0,0), radius=ps.p['r'][i]) for i in range(ps.p['n'])]        
            patch = ax.add_artist(PatchCollection(actors, offset_position='data', alpha=0.6, zorder=10))
            patch.set_array(np.random.rand(len(actors)))
            patches.append(patch)

        ps.update()        
        patches[0].set_offsets(ps.p['x'])
        return patches

    anim = animation.FuncAnimation(
        fig, 
        update_anim,  
        init_func=init_anim,
        interval=timestep * 1000,
        frames=frames,
        repeat=repeat,
        blit=True)

    return anim

if __name__ == '__main__':
    f = sdf.Halfspace(normal=[0, 1], d=-1.8) | sdf.Halfspace(normal=[1, 1], d=-1.8) | sdf.Halfspace(normal=[-1, 1], d=-1.8)

    with sdf.smoothness(10):
        f |= sdf.Circle(center=[0, 0.0], radius=0.5) & sdf.Halfspace(normal=[0.1, 1], d=0.3)

    g = sdf.GridSDF(f, samples=[100j, 100j])

    #with sdf.smoothness(20):
    #for i in range(10):
    #    with sdf.transform(angle=np.random.uniform(-0.5, 0.5), offset=np.random.uniform(-2, 2, size=2)):
    #        f |= sdf.Box(minc=[-0.2,-0.2], maxc=[0.2,0.2])

    def gravity(p, t):
        """Close to planet surface gravity."""
        return p['m'][:, np.newaxis] * np.array([0, -1]) 

    def grad(p, t, f=g):
        """Force field along gradients."""
        d, g = f(p['x'][:, 0], p['x'][:, 1], compute_gradient=True)
        return g * p['m'][:, np.newaxis]

    def create_particles(n):
        """Particle creator with some randomness."""

        p = {}
        p['n'] = n
        p['x'] = np.random.multivariate_normal([0, 1], [[0.05, 0],[0, 0.05]], n)
        p['v'] = np.random.multivariate_normal([0, 0], [[0.1, 0],[0, 0.1]], n)
        p['m'] = np.random.uniform(1, 10, size=n)
        p['r'] = p['m'] * 0.01
        p['cr'] = np.full(n, 0.6)
        p['cf'] = np.full(n, 0.4)
        
        return p

    # Create simulation
    n = 100
    ps = ParticleSimulator(g, lambda : create_particles(n), timestep=1/60)
    ps.force_generators += [gravity]

    # Plot result
    fig, ax = plt.subplots()
    plot_sdf(fig, ax, f, show_quiver=True, show_isolines='zero')
    anim = create_animation(fig, ax, ps, frames=500)
    plt.show()
