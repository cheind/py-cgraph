import cgraph as cg


if __name__ == '__main__':

    x = cg.Symbol('x')
    y = cg.Symbol('y')
    z = cg.Symbol('z')

    
    
    f = (x * y + 3) / (z - 2)
    v,d = cg.eval(f, x=3, y=4, z=4)

    print(f)
    print('f {}'.format(v))
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

    
    k = x*3-cg.pi
    m = f / k
    v,d = cg.eval(m, x=3, y=4, z=4)

    print(m)
    print('m {}'.format(v))
    print('dm/dx {}'.format(d[x]))
    print('dm/dy {}'.format(d[y]))
    print('dm/dz {}'.format(d[z]))

    """
    ((((x*y) + 3)/(z - 2))/((x*3) - pi))
    m 1.280211422067756
    dm/dx -0.3141868015256969
    dm/dy 0.2560422844135512
    dm/dz -0.64010571103387        
    """

    r = cg.Symbol('r')
    a = r * r * cg.pi
    print(a)
    v,d = cg.eval(a, r=3)
    
    print('a {}'.format(v))
    print('da/dr {}'.format(d[r]))