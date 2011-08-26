
def ldot(x,y):
    if isinstance(x, LinearTransform):
        # Ideally we could do something like
        # (y.T * x.T).T
        #
        # Except that one of the reasons the LinearTransform
        # is convenient is that it can include implicit reshapes.
        # Such things mess up the general transpose in this formula.
        raise NotImplementedError()
    elif isinstance(y, LinearTransform):
        return y.lmul(x)
    else:
        return theano.dot(x,y)

def ldot_outshape(x,y):
    if isinstance(x, LinearTransform):
        raise NotImplementedError()
    elif isinstance(y, LinearTransform):
        return y.lmul_outshape(x)
    else:
        raise NotImplementedError()

from linear import LinearTransform
from linear import MatrixMul
from linear import LConv
from linear import LConcat

