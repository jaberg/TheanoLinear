import unittest

import nose
import nose.plugins.attrib
import numpy

import theano
from theano.sandbox.cuda.var import float32_shared_constructor
from theano.sandbox.cuda import gpu_from_host

from .unshared_conv import FilterActs
from .unshared_conv import WeightActs
from .unshared_conv import ImgActs

from .gpu_unshared_conv import (
        GpuFilterActs,
        GpuWeightActs,
        GpuImgActs,
        )

import test_unshared_conv

def rand(shp, dtype):
    return numpy.random.rand(*shp).astype(dtype)

class TestGpuFilterActs(test_unshared_conv.TestFilterActs):
    """
    This class tests GpuWeightActs via the gradient of GpuFilterAct

    The correctness of GpuFilterActs is tested in TestMatchFilterActs
    """
    ishape = (1, 1, 4, 4, 2) # 2 4x4 greyscale images
    fshape = (2, 2, 1, 3, 3, 1, 16) # 5 3x3 filters at each location in a 2x2 grid
    module_stride = 1
    dtype = 'float32'
    mode = theano.compile.get_default_mode().including('gpu_opt',
            'fast_run', 'inplace').including('gpu_after_fusion',
                    'fast_run', 'inplace')

    def setUp(self):
        test_unshared_conv.TestFilterActs.setUp(self)
        self.gpu_op = GpuFilterActs(
                module_stride=self.module_stride,
                partial_sum=1)
        self.s_images = float32_shared_constructor(
                self.s_images.get_value())
        self.s_filters = float32_shared_constructor(
                self.s_filters.get_value())

    def test_gpu_shape(self):
        gpuout = self.gpu_op(self.s_images, self.s_filters)
        assert 'Cuda' in str(self.s_filters.type)
        f = theano.function([], gpuout)
        outval = f()
        assert outval.shape == (
                self.fshape[-2], self.fshape[-1],
                self.fshape[0], self.fshape[1],
                self.ishape[-1])

    def test_insert_gpu_filter_acts(self):
        out = self.op(self.s_images, self.s_filters)
        f = self.function([], out)
        assert isinstance(
                f.maker.env.toposort()[0].op,
                GpuFilterActs)

    def test_gpu_op_eq(self):
        assert GpuFilterActs(1, 1) == GpuFilterActs(1, 1)
        assert not (GpuFilterActs(1, 1) != GpuFilterActs(1, 1))
        assert (GpuFilterActs(1, 2) != GpuFilterActs(1, 1))
        assert (GpuFilterActs(2, 1) != GpuFilterActs(1, 1))
        assert GpuFilterActs(2, 1) != None


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


if 0:
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


@nose.plugins.attrib.attr('slow')
def test_fuzz_conv_GpuFilterActs(dtype='float32'):
    ishape = (2, 4, 4, 3)
    fshape = (5, 2, 2, 3)

    s_imgs = theano.shared(rand(ishape, dtype))
    s_filters = theano.shared(rand(fshape, dtype))
    sf_imgs = theano.shared(rand((2, 2, 2, 2, 2), dtype))
    sf_filters = theano.shared(rand((1, 1, 2, 2, 2, 2, 2), dtype))
    s_out_c = theano.shared(rand(ishape, dtype))
    s_out_f = theano.shared(rand((2, 2, 2, 2, 2), dtype))

    rng = numpy.random.RandomState(234)

    for i in range(10):
        n_imgs = rng.randint(16) + 1
        channels = rng.randint(256) + 1
        rows = rng.randint(512) + 1
        cols = rows
        frows = rng.randint(17) + 1
        fcols = frows
        n_filters = rng.randint(256) + 16
        n_filters -= n_filters % 16

        c_imgs = rand((n_imgs, channels, rows, cols), dtype=dtype)
        c_filters = rand((n_filters, channels, frows, fcols), dtype=dtype)

        print 'SHAPES', c_imgs.shape, c_filters.shape

        s_imgs.set_value(c_imgs)
        s_filters.set_value(c_filters)
        sf_imgs.set_value(c_imgs.transpose(1, 2, 3, 0)[None,:,:,:,:])
        sf_filters.set_value(c_filters.transpose(1, 2, 3, 0)[None, None,:,:,:,None,:])

        conv_op = theano.tensor.nnet.conv.ConvOp(
                imshp=(channels, rows, cols),
                kshp=(frows, fcols),
                bsize=n_imgs,
                nkern=n_filters)

        def fa_op(ii, ff):
            op = GpuFilterActs(module_stride=1, partial_sum=1, conv=True)
            return op(ii, ff)# gpu_from_host(ii), gpu_from_host(ff))

        f = theano.function([], [], updates={
            s_out_c: conv_op(s_imgs, s_filters),
            s_out_f: fa_op(sf_imgs, sf_filters),
            })
        f()
        out_c = s_out_c.get_value()
        out_f = s_out_f.get_value()
        print out_c.shape
        print out_c.mean()
        print out_f.shape
        print out_f.mean()
        assert out_c.shape == out_f.shape
        assert numpy.allclose(out_c, out_f)

