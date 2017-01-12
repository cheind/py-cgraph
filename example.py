import cgraph as cg


if __name__ == '__main__':

    x = cg.Symbol('x')
    y = cg.Symbol('y')
    z = cg.Symbol('z')

    
    
    f = (x * y + 3) / (z - 2)
    print(f)
    print('f {}'.format(f.eval({x:3, y:4, z:4})))
    
    d = f.ndiff({x:3, y:4, z:4})
    print('df/dx {}'.format(d[x]))
    print('df/dy {}'.format(d[y]))
    print('df/dz {}'.format(d[z]))
    
    """
    (((x*y) + 3)/(z - 2))
    f 7.5
    df/dx 2.0
    df/dy 1.5
    df/dz -3.75
    """
