import numpy as np

import cgraph as cg
import cgraph.ops.addition
import matplotlib.pyplot as plt

def generate_points(n, w0, w1):

    x = np.linspace(0, 10, n)
    y = x * w0 + w1 + np.random.normal(scale=0.1, size=n)
    return np.vstack((x,y))

def sum_residuals_squared(w0, w1, xy):

    residuals = []
    for i in range(xy.shape[1]):
        r = w0 * xy[0,i] + w1 - xy[1,i]
        residuals.append(r**2)

    return cgraph.ops.addition.sym_sum(residuals)

def least_squares(xy):

    A = np.ones((xy.shape[1], 2))    
    A[:,0] = xy[0,:].T
    b = xy[1,:].T

    return np.dot(np.dot(np.linalg.inv(np.dot(A.T,A)), A.T), b)
    

if __name__ == '__main__':

    # Parameters of ideal line
    k = 0.8
    d = 2.0

    samples = generate_points(20, k, d)
    
    w0 = cg.Symbol('w0')
    w1 = cg.Symbol('w1')    
    
    f = sum_residuals_squared(w0, w1, samples)

    guess = {w0: 0., w1: 1.1}
    step = 0.001

    for i in range(400):      
        df = f.ndiff(guess)

        guess[w0] -= step * df[w0]
        guess[w1] -= step * df[w1]

        print('Error {}'.format(f.eval(guess)))

    
    fit = least_squares(samples)
    plt.plot([0, 10], [0*fit[0]+fit[1], 10*fit[0]+fit[1]], color='r', linestyle='-', linewidth=1, label='least squares')
    
    plt.scatter(samples[0,:], samples[1,:])
    plt.plot([0, 10], [0*k+d, 10*k+d], color='k', linestyle=':', linewidth=1, label='ground truth')
    
    plt.plot([0, 10], [0*guess[w0]+guess[w1], 10*guess[w0]+guess[w1]], color='g', linestyle='-', linewidth=1, label='optimized')
    plt.legend()
    plt.show()


