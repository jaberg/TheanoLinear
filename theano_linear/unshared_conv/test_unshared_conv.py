
import unittest

import numpy

import theano
from theano.sandbox.cuda.var import float32_shared_constructor
from theano.tests.unittest_tools import verify_grad

from .unshared_conv import FilterActs
from .unshared_conv import WeightActs
from .unshared_conv import ImgActs

def rand(shp, dtype):
    return numpy.random.rand(*shp).astype(dtype)

def assert_linear(f, pt):
    t = theano.tensor.scalar(dtype=pt.dtype)
    ptlike = theano.shared(rand(
        pt.get_value(borrow=True).shape,
        dtype=pt.dtype))
    out = f(pt)
    out2 = f(pt * t)
    out3 = f(ptlike) + out
    out4 = f(pt + ptlike)

    f = theano.function([t], [out * t, out2, out3, out4],
            allow_input_downcast=True)
    outval, out2val, out3val, out4val = f(3.6)
    assert numpy.allclose(outval, out2val)
    assert numpy.allclose(out3val, out4val)


class TestFilterActs(unittest.TestCase):
    # 2 4x4 greyscale images
    ishape = (1, 1, 4, 4, 2)
    # 5 3x3 filters at each location in a 2x2 grid
    fshape = (2, 2, 1, 3, 3, 1, 5)
    module_stride = 1
    dtype = 'float64'

    def setUp(self):
        self.fa = FilterActs(self.module_stride)
        self.s_images = theano.shared(rand(self.ishape, self.dtype))
        self.s_filters = theano.shared(rand(self.fshape, self.dtype))

    def test_type(self):
        out = self.fa(self.s_images, self.s_filters)
        assert out.dtype == self.dtype
        assert out.ndim == 5

        f = theano.function([], out)
        outval = f()
        assert len(outval.shape) == len(self.ishape)
        assert outval.dtype == self.s_images.get_value(borrow=True).dtype

    def test_linearity_images(self):
        assert_linear(
                lambda imgs: self.fa(imgs, self.s_filters),
                self.s_images)

    def test_linearity_filterss(self):
        assert_linear(
                lambda fts: self.fa(self.s_images, fts),
                self.s_filters)

    def test_shape(self):
        out = self.fa(self.s_images, self.s_filters)
        f = theano.function([], out)
        outval = f()
        assert outval.shape == (self.fshape[-2],
                self.fshape[-1],
                self.fshape[0], self.fshape[1],
                self.ishape[-1])

    def test_grad(self):
        try:
            verify_grad(self.fa,
                    [self.s_images.get_value(),
                        self.s_filters.get_value()])
        except verify_grad.E_grad, e:
            print e.num_grad.gf
            print e.analytic_grad
            raise

    def test_dtype_mismatch(self):
        self.assertRaises(TypeError,
                self.fa,
                theano.tensor.cast(self.s_images, 'float32'),
                theano.tensor.cast(self.s_filters, 'float64'))
        self.assertRaises(TypeError,
                self.fa,
                theano.tensor.cast(self.s_images, 'float64'),
                theano.tensor.cast(self.s_filters, 'float32'))


class TestFilterActsF32(TestFilterActs):
    dtype = 'float32'


class TestWeightActs(unittest.TestCase):
    # 1 5x5 6-channel image (2 groups of 3 channels)
    ishape = (6, 3, 5, 5, 1)
    hshape = (6, 4, 2, 2, 1)
    fshape = (2, 2, 3, 2, 2, 6, 4)
    module_stride = 2
    dtype = 'float64'

    frows = property(lambda s: s.fshape[3])
    fcols = property(lambda s: s.fshape[4])

    def setUp(self):
        self.op = WeightActs(self.module_stride)
        self.s_images = theano.shared(rand(self.ishape, self.dtype))
        self.s_hidacts = theano.shared(rand(self.hshape, self.dtype))

    def test_type(self):
        out = self.op(self.s_images, self.s_hidacts, self.frows, self.fcols)
        assert out.dtype == self.dtype
        assert out.ndim == 7
        f = theano.function([], out)
        outval = f()
        assert outval.shape == self.fshape
        assert outval.dtype == self.dtype

    def test_linearity_images(self):
        def f(images):
            return self.op(images, self.s_hidacts, self.frows, self.fcols)
        assert_linear(f, self.s_images)

    def test_linearity_hidacts(self):
        def f(hidacts):
            return self.op(self.s_images, hidacts, self.frows, self.fcols)
        assert_linear(f, self.s_hidacts)
