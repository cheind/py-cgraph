
def z(x):
    """Returns NAN if input is zero."""
    return x if x != 0 else float('nan')

def n(x):
    """Returns NAN if input is less zero."""
    return x if x >= 0 else float('nan')

def zn(x):
    """Returns NAN if input is less than or equal zero."""
    return x if x > 0 else float('nan')

