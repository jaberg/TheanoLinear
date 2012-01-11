
import unittest

import numpy

from .linear import LinearTransform
from .linear import dot
from .linear import dot_shape
from .linear import dot_shape_from_shape


class ReshapeBase(LinearTransform):
    def __init__(self, from_shp, to_shp):
        LinearTransform.__init__(self, [])
        self._from_shp = from_shp
        self._to_shp = to_shp

    def row_shape(self):
        return self._to_shp

    def col_shape(self):
        return self._from_shp


class ReshapeL(ReshapeBase):
    def lmul(self, x, T=False):
        # dot(x, A) or dot(x, A.T)
        if T:
            return x.reshape(x.shape[0:1] + self._from_shp)
        else:
            return x.reshape(x.shape[0:1] + self._to_shp)


class ReshapeR(ReshapeBase):
    def rmul(self, x, T=False):
        # dot(A, x) or dot(A.T, x)
        if T:
            return x.reshape(x.shape[0:1] + self._from_shp)
        else:
            return x.reshape(x.shape[0:1] + self._to_shp)


class SelfTestMixin(object):
    """
    Generic tests that assert the self-consistency of LinearTransform
    implementations.

    """

    def test_shape_xl_A(self):
        xl_A = dot(self.xl, self.A)
        assert xl_A.shape == dot_shape(self.xl, self.A)

    def test_shape_A_xr(self):
        A_xr = dot(self.A, self.xr)
        assert A_xr.shape == dot_shape(self.A, self.xr)

    def test_shape_xrT_AT(self):
        xrT_AT = dot(self.A.T.transpose_left(self.xr), self.A.T)
        assert xrT_AT.shape == dot_shape_from_shape(
                self.A.T.transpose_left_shape(self.xr.shape), self.A.T)

    def test_shape_AT_xlT(self):
        AT_xlT = dot(self.A.T, self.A.T.transpose_right(self.xl))
        assert AT_xlT.shape == dot_shape_from_shape(self.A.T,
                A.T.transpose_right_shape(self.xr.shape))


class TestReshapeL(SelfTestMixin):
    def setUp(self):
        self.xl = numpy.random.randn(4, 3, 2)  # for left-mul
        self.xr = numpy.random.randn(6, 5)     # for right-mul
        self.A = ReshapeL((3, 2), (6,))
        self.xl_A_shape = (4, 6)

    def test_xl_A_value(self):
        xl_A = dot(self.xl, self.A)
        assert numpy.all(xl_A == self.xl.reshape(xl_A.shape))

