"""Uses auto differentiation to fit a line to a set of 2 dimensional points."""

import numpy as np

import cgraph as cg
import matplotlib.pyplot as plt

def generate_points(n, k, d):
    """Returns noisy random points on line."""

    x = np.linspace(0, 10, n)
    y = x * k + d + np.random.normal(scale=0.1, size=n)
    return np.vstack((x,y))

def sum_residuals_squared(w, xy):
    """Returns the symbolic computational graph for the objective minimization
    
    In particular this builds the computational graph that computes the average
    squared algebraic distance between the given samples and the line expressed 
    through parameters w0 and w1.
    """
    n = xy.shape[1]
    residuals = []
    for i in range(n):
        r = w[0] * xy[0,i] + w[1] - xy[1,i]
        residuals.append(cg.sym_sqr(r))

    return cg.sym_sum(residuals) / n

def least_squares(xy):
    """Returns the line parameters through ordinary least squares regression."""

    A = np.ones((xy.shape[1], 2))    
    A[:,0] = xy[0,:].T
    b = xy[1,:].T

    return np.dot(np.dot(np.linalg.inv(np.dot(A.T,A)), A.T), b)


def steepest_descent(f, w, guess):
    print('Entering steepest descent')

    lam = 0.02

    for i in range(200):
        # Auto-diff, could also do f.sdiff() + eval for symbolic diff.        
        df = cg.numeric_gradient(f, guess)

        guess[w[0]] -= lam * df[w[0]]
        guess[w[1]] -= lam * df[w[1]]

        print('Error {}'.format(cg.value(f, guess)))

    return guess

def newton_descent(f, w, guess):
    print('Entering Newton descent')

    d1 = cg.symbolic_gradient(f)        # gives df/dw0, df/dw1
    d2w0 = cg.symbolic_gradient(d1[w[0]]) # gives ddf/dw0dw0, ddf/dw0dw1,
    d2w1 = cg.symbolic_gradient(d1[w[1]]) # gives ddf/dw1dw1, ddf/dw1dw0,

    def nhessian(guess):
        h = np.zeros((2,2))
        h[0,0] = cg.value(d2w0[w[0]], guess)
        h[0,1] = cg.value(d2w0[w[1]], guess)
        h[1,0] = cg.value(d2w1[w[0]], guess)
        h[1,1] = cg.value(d2w1[w[1]], guess)
        return h

    def ngrad(guess):
        g = np.zeros((2,1))
        g[0, 0] = cg.value(d1[w[0]], guess)
        g[1, 0] = cg.value(d1[w[1]], guess)
        return g

    # Single step is enough, since our objective function
    # is of quadric shape.
    step = np.linalg.inv(nhessian(guess)).dot(ngrad(guess))
    guess[w[0]] -= step[0,0]
    guess[w[1]] -= step[1,0]

    print('Error {}'.format(cg.value(f, guess)))

    return guess

if __name__ == '__main__':

    # Parameters of ideal line
    k = 0.8
    d = 2.0

    # Noisy line samples
    samples = generate_points(20, k, d)
    
    # The parameters we optimize for
    w = [
        cg.Symbol('w0'),
        cg.Symbol('w1')
    ]   
    
    # Build the computational graph
    f = sum_residuals_squared(w, samples)
    
    s_sd = steepest_descent(f, w, {w[0]: 0.4, w[1]: 1.1})
    s_nd = newton_descent(f, w, {w[0]: 0.4, w[1]: 1.1})
    s_fit = least_squares(samples)

    # Draw results
    plt.plot([0, 10], [0*s_fit[0]+s_fit[1], 10*s_fit[0]+s_fit[1]], color='r', linestyle='-', label='Least Squares')
    plt.plot([0, 10], [0*s_sd[w[0]]+s_sd[w[1]], 10*s_sd[w[0]]+s_sd[w[1]]], color='g', linestyle='-', label='Steepest Descent')
    plt.plot([0, 10], [0*s_nd[w[0]]+s_nd[w[1]], 10*s_nd[w[0]]+s_nd[w[1]]], color='b', linestyle='-', label='Newton Descent')
    plt.plot([0, 10], [0*k+d, 10*k+d], color='k', linestyle=':', label='Ground Truth')
    
    plt.scatter(samples[0,:], samples[1,:])
    plt.legend()
    plt.show()
