
import pytest
from pytest import approx

def checkf(f, fargs, value=None, ngrad=None):
    __tracebackhide__ = True

    kargs = {}
    for k,v in fargs.items():
        kargs[k.name] = v
    
    v = f.eval(**kargs)
    if value != approx(f.eval(**kargs)):
        pytest.fail("""Function VALUE check failed
        f: {}
        expected value of {} - received {}""".format(f, value, v))

    ng = f.ndiff(**kargs)
    sg = f.sdiff()

    for k in fargs.keys():
        if ngrad[k] != approx(ng[k]):
            pytest.fail("""Function NUMERIC GRAD check failed
            f: {}, 
            df/d{}
            expected value of {} - received {}""".format(f, k, ngrad[k], ng[k]))

        if ngrad[k] != approx(sg[k].eval(**kargs)):
            pytest.fail("""Function SYMBOLIC GRAD check failed
            f: {}, 
            df/d{}: {},
            expected value of {} - received {}""".format(f, k, sg[k], ngrad[k], sg[k].eval(**kargs)))