import cgraph as cg
import cgraph.simplify
import cgraph.plot

if __name__ == '__main__':

    x = cg.Symbol('x')
    y = cg.Symbol('y')
    z = cg.Symbol('z')
    
    f = (x * y + 3) / (z - 2)
    cgraph.plot.plot(f, 'graph.gv')

