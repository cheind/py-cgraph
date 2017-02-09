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

class ParticleSimulation:
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

    def __init__(self, f, n=100, timestep=1/30):

        self.p = None        
        self.dt = timestep
        self.n = n
        
        self.sdf = f
        self.force_generators = []
        self.collection = None
        
    def reset(self, ax):
        """Reset simulation."""

        self.p = self.create_particles()
        self.p['f'] = np.zeros((self.p['n'], 2))

        if self.collection:
            self.collection.remove()

        actors = [plt.Circle((0,0), radius=self.p['r'][i]) for i in range(self.n)]        
        self.collection = ax.add_artist(PatchCollection(actors, offset_position='data', alpha=0.6, zorder=10))
        self.collection.set_array(np.random.rand(len(actors)))

        self.t = 0        
        self.tacc = 0.
        self.current_wall_time = time.time()

    def create_particles(self):
        p = {}
        
        p['n'] = self.n
        p['x'] = np.random.multivariate_normal([0, 1.5], [[0.05, 0],[0, 0.05]], n)
        p['v'] = np.random.multivariate_normal([0, 0], [[0.1, 0],[0, 0.1]], n)
        p['m'] = np.random.uniform(1, 10, size=self.n)
        p['r'] = p['m'] * 0.01
        p['cr'] = np.full(n, 0.6)
        p['cf'] = np.full(n, 0.3)

        return p

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

    def integrate(self, x, v, dx, dv, t, dt):
        """Evolve in ODE in time by performing a single explicit euler step."""
    
        return x + dx * dt, v + dv * dt


    #@timeit
    def update(self, ax, use_wall_time=True):
        """Update the simulation.
        
        The ParticleSimulator uses a fixed timestepping scheme.
        In order to determine the number of simulation rounds
        required, ParticleSimulator measures the elapsed time
        since last invocation. When the accumulated time is
        greater than the timestep it performs one or more steps.
        """

        if use_wall_time:
            new_wall_time = time.time()
            frame_time = new_wall_time - self.current_wall_time
            self.current_wall_time = new_wall_time
            self.tacc += frame_time

            while self.tacc >= self.dt:
                self.advance()
                self.tacc -= self.dt
                self.t += self.dt
        else:
            self.advance()
            self.t += self.dt

        self.collection.set_offsets(self.p['x'])
        return self.collection,
                
    
    def advance(self):
        """Advance time by one timestep."""

        # Compute dynamics and state at t + dt
        xcur = self.p['x']
        vcur = self.p['v']

        dx, dv = self.dynamics()
        xnew, vnew = self.integrate(xcur, vcur, dx, dv, self.t, self.dt)
        
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

            # The gradient should already be close to unit length, but
            # we normalize here to avoid numeric effects.
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

def create_animation(fig, ax, simulation, bounds=[(-2,2), (-2,2)], frames=500, timestep=1/30, repeat=True, use_wall_time=True):
    """Create a matplotlib animation involving a particle simulation."""

    state = {}

    def init_anim():
        sdf.setup_plot_axes(ax, bounds)      
        state['reset'] = True                        
        return []

    def update_anim(i):
        if 'reset' in state:
            # We initialize the circles in here. It seems like matplotlib keeps a static image
            # of all circles at (0,0) when calling the same method inside init_anim. Also, we need a
            # new circle collection when an animation repeats, because radii of circles might have 
            # changed during ps.reset()            
            simulation.reset(ax)
            del state['reset']

        return simulation.update(ax, use_wall_time=use_wall_time)
        
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

    # Some random boxes
    #for i in range(10):
    #    with sdf.transform(angle=np.random.uniform(-0.8, 0.8), offset=np.random.uniform(-2, 2, size=2)):
    #        f |= sdf.Box(minc=np.random.uniform(-0.3, -0.1, size=2), maxc=np.random.uniform(0.1, 0.3, size=2))    

    #with sdf.smoothness(20):
    #for i in range(10):
    #    with sdf.transform(angle=np.random.uniform(-0.5, 0.5), offset=np.random.uniform(-2, 2, size=2)):
    #        f |= sdf.Box(minc=[-0.2,-0.2], maxc=[0.2,0.2])

    def gravity(p, t):
        """Close to planet surface gravity."""
        return p['m'][:, np.newaxis] * np.array([0, -1]) 

    def grad(p, t, f):
        """Force field along gradients."""
        d, g = f(p['x'][:, 0], p['x'][:, 1], compute_gradient=True)
        return g * p['m'][:, np.newaxis]

    # Discretize signed distance function using a grid for fast lookup.
    # Note, if the number of samples is too small you might see particles get stuck
    # during narrow places.
    bounds=[(-2,2), (-2,2)]
    g = sdf.GridSDF(f, bounds=bounds, samples=[200j, 200j])

    # Create simulation
    n = 100
    ps = ParticleSimulation(g, n=n, timestep=1/60)
    #ps.force_generators += [gravity, lambda p, t: grad(p, t, g)]
    ps.force_generators += [gravity]

    # Plot result
    fig, ax = plt.subplots()
    fig.set_size_inches(1280/fig.dpi, 720/fig.dpi)
    sdf.plot_sdf(fig, ax, g, bounds=bounds, show_quiver=True, show_isolines='zero')
    anim = create_animation(fig, ax, ps, bounds=bounds, frames=400)

    #anim.save('test.mp4',fps=30, dpi=400)
    
    plt.show()
