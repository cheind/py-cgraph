
import pytest
from pytest import approx

def checkf(f, fargs, value=None, ngrad=None):
    __tracebackhide__ = True
   
    v = f.eval(fargs)
    if value != approx(v):
        pytest.fail("""Function VALUE check failed
        f: {}
        expected value of {} - received {}""".format(f, value, v))

    ng = f.ndiff(fargs)
    sg = f.sdiff()

    for k in fargs.keys():
        if ngrad[k] != approx(ng[k]):
            pytest.fail("""Function NUMERIC GRAD check failed
            f: {}, 
            df/d{}
            expected value of {} - received {}""".format(f, k, ngrad[k], ng[k]))

        if ngrad[k] != approx(sg[k].eval(fargs)):
            pytest.fail("""Function SYMBOLIC GRAD check failed
            f: {}, 
            df/d{}: {},
            expected value of {} - received {}""".format(f, k, sg[k], ngrad[k], sg[k].eval(fargs)))