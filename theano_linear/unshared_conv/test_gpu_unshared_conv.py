import unittest

import numpy

import theano
from theano.sandbox.cuda.var import float32_shared_constructor

from .unshared_conv import FilterActs
from .unshared_conv import WeightActs
from .unshared_conv import ImgActs

from .gpu_unshared_conv import GpuFilterActs
from .gpu_unshared_conv import GpuWeightActs
#from .gpu_unshared_conv import GpuFilterActs

class TestGpuFilterActs(unittest.TestCase):
    """
    This class tests GpuWeightActs via the gradient of GpuFilterAct

    The correctness of GpuFilterActs is tested in TestMatchFilterActs
    """
    ishape = (1, 1, 4, 4, 2) # 2 4x4 greyscale images
    fshape = (2, 2, 1, 3, 3, 1, 16) # 5 3x3 filters at each location in a 2x2 grid
    module_stride = 1
    dtype = 'float32'


    def setUp(self):
        self.gfa = GpuFilterActs(self.module_stride)
        self.gpu_images = float32_shared_constructor(
                numpy.random.rand(*self.ishape).astype(self.dtype))
        self.gpu_filters = float32_shared_constructor(
                numpy.random.rand(*self.fshape).astype(self.dtype))

    def test_shape(self):
        gpuout = self.gfa(self.gpu_images, self.gpu_filters)
        f = theano.function([], gpuout)
        outval = f()
        assert outval.shape == (
                self.fshape[-2], self.fshape[-1],
                self.fshape[0], self.fshape[1],
                self.ishape[-1])

    def test_gradient_type(self):
        gpuout = self.gfa(self.gpu_images, self.gpu_filters)
        dimages, dfilters = self.gfa.grad(gpuout.owner.inputs, [gpuout])

        assert dfilters.broadcastable == (False,) * 7


class TestGpuWeightActs(unittest.TestCase):
    """
    """
    ishape = (1, 1, 4, 4, 2) # 2 4x4 greyscale images
    hshape = (1, 16, 2, 2, 2)
    fshape = (2, 2, 1, 3, 3, 1, 16) # 5 3x3 filters at each location in a 2x2 grid
    frows = 3
    fcols = 3
    module_stride = 1
    partial_sum = 1
    dtype = 'float32'

    def setUp(self):
        self.gwa = GpuWeightActs(
                module_stride=self.module_stride,
                partial_sum=self.partial_sum)
        self.gpu_images = float32_shared_constructor(
                numpy.random.rand(*self.ishape).astype(self.dtype))
        self.gpu_hidact = float32_shared_constructor(
                numpy.random.rand(*self.hshape).astype(self.dtype))

    def test_shape(self):
        dfilters = self.gwa(self.gpu_images, self.gpu_hidact,
                self.frows, self.fcols)
        f = theano.function([], dfilters)
        outval = f()
        assert outval.shape == self.fshape


class TestMatchFilterActs(unittest.TestCase):
    def setUp(self):
        numpy.random.seed(77)

    def run_match(self, images, filters, module_stride, retvals=False):

        gfa = GpuFilterActs(module_stride)
        fa = FilterActs(module_stride)

        gpu_images = float32_shared_constructor(images)
        gpu_filters = float32_shared_constructor(filters)
        cpu_images = theano.shared(images)
        cpu_filters = theano.shared(filters)

        gpu_out = gfa(gpu_images, gpu_filters)
        cpu_out = fa(cpu_images, cpu_filters)

        f = theano.function([], [cpu_out, gpu_out])
        cpuval, gpuval = f()
        gpuval = numpy.asarray(gpuval)

        if retvals:
            return cpuval, gpuval
        else:
            #print 'run_match: cpu shape', cpuval.shape
            #print 'run_match: gpu shape', gpuval.shape
            assert cpuval.shape == gpuval.shape
            assert numpy.allclose(cpuval, gpuval)

    def run_match_shape(self, ishape, fshape, module_stride, dtype='float32'):
        return self.run_match(
            images=numpy.random.rand(*ishape).astype(dtype),
            filters=numpy.random.rand(*fshape).astype(dtype),
            module_stride=module_stride)

    def test_small_random(self):
        self.run_match_shape(
            ishape = (1, 1, 4, 4, 2),
            fshape = (2, 2, 1, 3, 3, 1, 16),
            module_stride = 1)

    def test_small_random_colors(self):
        self.run_match_shape(
            ishape = (1, 6, 4, 4, 2),
            fshape = (2, 2, 6, 3, 3, 1, 16),
            module_stride = 1)

    def test_small_random_groups(self):
        self.run_match_shape(
            ishape = (5, 6, 4, 4, 2),
            fshape = (2, 2, 6, 3, 3, 5, 16),
            module_stride = 1)

    def test_small_random_module_stride(self):
        self.run_match_shape(
            ishape = (4, 6, 5, 5, 1),
            fshape = (2, 2, 6, 3, 3, 4, 16),
            module_stride = 2)

    def test_med_random_module_stride(self):
        self.run_match_shape(
            ishape = (4, 6, 32, 32, 1),
            fshape = (12, 12, 6, 3, 3, 4, 16),
            module_stride = 2)


    def _blah_topcorner_filter1(self):
        ishape = (1, 1, 4, 4, 2)
        fshape = (2, 2, 1, 3, 3, 1, 16)
        images = numpy.random.rand(*ishape).astype('float32')
        filters = numpy.random.rand(*fshape).astype('float32')
        filters *= 0
        filters[0,0,0,0,0,0,0] = 1
        self.run_match(images, filters, 1)

    def _blah_botcorner_filter1(self):
        ishape = (1, 1, 4, 4, 2)
        fshape = (2, 2, 1, 3, 3, 1, 16)
        images = numpy.random.rand(*ishape).astype('float32')
        filters = numpy.random.rand(*fshape).astype('float32')
        filters *= 0
        filters[1,1,0,0,0,0,0] = 1
        cpuval, gpuval = self.run_match(images, filters, 1, retvals=True)
        print images
        print cpuval[:, :, 1, 1, :]
        print gpuval[:, :, 1, 1, :]

