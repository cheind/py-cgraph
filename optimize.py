"""Uses auto differentiation to fit a line to a set of 2 dimensional points."""

import numpy as np

import cgraph as cg
from cgraph.ops.addition import sym_sum
import matplotlib.pyplot as plt

def generate_points(n, k, d):
    """Returns noisy random points on line."""

    x = np.linspace(0, 10, n)
    y = x * k + d + np.random.normal(scale=0.1, size=n)
    return np.vstack((x,y))

def sum_residuals_squared(w0, w1, xy):
    """Returns the symbolic computational graph for the objective minimization
    
    In particular this builds the computational graph that computes the average
    squared algebraic distance between the given samples and the line expressed 
    through parameters w0 and w1.
    """
    n = xy.shape[1]
    residuals = []
    for i in range(n):
        r = w0 * xy[0,i] + w1 - xy[1,i]
        residuals.append(r**2)

    return sym_sum(residuals) / n

def least_squares(xy):
    """Returns the line parameters through ordinary least squares regression."""

    A = np.ones((xy.shape[1], 2))    
    A[:,0] = xy[0,:].T
    b = xy[1,:].T

    return np.dot(np.dot(np.linalg.inv(np.dot(A.T,A)), A.T), b)
    

if __name__ == '__main__':

    # Parameters of ideal line
    k = 0.8
    d = 2.0

    # Noisy line samples
    samples = generate_points(40, k, d)
    
    # The parameters we optimize for
    w0 = cg.Symbol('w0')
    w1 = cg.Symbol('w1')    
    
    # Build the computational graph
    f = sum_residuals_squared(w0, w1, samples)
    
    # Initial guess for gradient descent
    guess = {w0: 0., w1: 1.1}
    step = 0.01

    for i in range(200):
        # Auto-diff, could also do f.sdiff() + eval for symbolic diff.        
        df = f.ndiff(guess)

        guess[w0] -= step * df[w0]
        guess[w1] -= step * df[w1]

        print('Error {}'.format(f.eval(guess)))

    # Draw the line fitted by least squares
    fit = least_squares(samples)
    plt.plot([0, 10], [0*fit[0]+fit[1], 10*fit[0]+fit[1]], color='r', linestyle='-', label='least squares')
    
    # Draw the ground truth line
    plt.scatter(samples[0,:], samples[1,:])
    plt.plot([0, 10], [0*k+d, 10*k+d], color='k', linestyle=':', label='ground truth')
    
    # Draw the line found through optimization
    plt.plot([0, 10], [0*guess[w0]+guess[w1], 10*guess[w0]+guess[w1]], color='g', linestyle='-', label='optimized')
    plt.legend()
    plt.show()


