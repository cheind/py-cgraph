import pytest
import numpy as np
import cgraph as cg

def checkf(f, fargs, value=None, ngrad=None, with_sgrad=True):
    __tracebackhide__ = True
   
    v = cg.value(f, fargs)
    if not all(np.isclose(value, v)):
        pytest.fail("""Function VALUE check failed
        f: {}
        expected value of {} - received {}""".format(f, value, v))

    if ngrad is not None:
        ng = cg.numeric_gradient(f, fargs)    
        for k in fargs.keys():
            if not all(np.isclose(ngrad[k], ng[k])):
                pytest.fail("""Function NUMERIC GRAD check failed
                f: {}, 
                df/d{}
                expected value of {} - received {}""".format(f, k, ngrad[k], ng[k]))

    if ngrad is not None and with_sgrad:
        ng = cg.numeric_gradient(f, fargs)
        sg = cg.symbolic_gradient(f) 
        for k in fargs.keys():
            if not all(np.isclose(ngrad[k], cg.value(sg[k], fargs))):
                pytest.fail("""Function SYMBOLIC GRAD check failed
                f: {}, 
                df/d{}: {},
                expected value of {} - received {}""".format(f, k, sg[k], ngrad[k], cg.value(sg[k], fargs)))