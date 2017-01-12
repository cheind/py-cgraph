import cgraph as cg
import cgraph.simplify

if __name__ == '__main__':

    x = cg.Symbol('x')
    y = cg.Symbol('y')
    z = cg.Symbol('z')

    
    
    f = (x * y + 3) / (z - 2)
    d = f.sdiff()

    print(d[x])
    print(cgraph.simplify.simplify(d[x]))