import cgraph as cg

@cg.wrap_args
def circle(x, y, cx=0., cy=0., r=1.):
    """Return the signed distance function for a circle."""

    return cg.sym_sqrt((cx - x)**2 + (cy - y)**2) - r


def union(x, y):
    return cg.sym_min(x, y)
