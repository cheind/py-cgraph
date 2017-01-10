import cgraph as cg


if __name__ == '__main__':

    x = cg.Symbol('x')
    y = cg.Symbol('y')
    z = cg.Symbol('z')

    f = (x * y + 3) / (z - 2)
    v,d = cg.eval(f, x=3, y=4, z=4)

    print('f {}'.format(v))
    print('df/dx {}'.format(d[x]))
    print('df/dy {}'.format(d[y]))
    print('df/dz {}'.format(d[z]))

    """
    k = x*3-y
    v,d = k.eval(x=3, y=4)
    print(v)
    print(d)

    m = f / k
    v,d = m.eval(x=3, y=4, z=4)
    print(v)
    print(d)
    """