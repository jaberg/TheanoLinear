import logging
logger = logging.getLogger(__name__)
import copy
import nose
import nose.plugins.attrib
import numpy as np
#from scipy.ndimage.filters import correlate, convolve

import theano
from theano.sandbox.cuda.basic_ops import as_cuda_ndarray_variable, gpu_contiguous

import pycuda
#import pycuda.autoinit
from pycuda import driver
def init_cuda(dev_id=0):
    driver.init()
    logger.info( "GPU Device listing")
    for i in range(driver.Device.count()):
        device = driver.Device(i)
        logger.info( "Device %i: %s %s" % (i,  device.name(),
            device.compute_capability()))
    device = driver.Device(dev_id)
    logger.info("Using: %s" % device.name())
    return device

device0 = init_cuda()

class CudaContext(object):
    def __init__(self, device):
        self.device = device
    def __enter__(self):
        self.context = self.device.make_context()
        return self.context
    def __exit__(self, *args):
        #print "MEM INFO", pycuda.driver.mem_get_info()
        self.context.pop()
        self.context.detach()

def with_cuda_context(f):
    def wrapper(*args, **kwargs):
        with CudaContext(device0) as context:
            return f(context=context, *args, **kwargs)
    wrapper.__name__ = f.__name__
    return wrapper


from fbconv3_cuda import FilterOp, Input, Output, Filterbank


class GCG_FBCorr(theano.Op):
    def __init__(self, img_shp, ker_shp, context, **fop_kwargs):
        # XXX: delay this construction until make_thunk
        self.fop_kwargs = copy.deepcopy(fop_kwargs)
        N, H, W, C = img_shp
        assert N == 1
        self.img_shp = (H, W, C)
        self.ker_shp = ker_shp # n, rows, cols, channels
        self.out_shp = (
                1 + H - ker_shp[1],
                1 + W - ker_shp[2],
                ker_shp[0])
        self.context = context

    def __eq__(self, other):
        return self is other  # XXX look at attributes

    def __hash__(self):
        return id(self)

    def __str__(self):
        return 'GCG_FBCorr{img_shp=%s,ker_shp=%s}' % (
                str(self.img_shp),
                str(self.ker_shp))

    def make_node(self, image, filters):
        return theano.Apply(self, [image, filters], [image.type()])

    def __call__(self, image, filters):
        if 0:
            contig_image = gpu_contiguous(image)
            contig_filters = gpu_contiguous(filters)
        else:
            contig_image = image
            contig_filters = filters
        return self.make_node(contig_image, contig_filters).outputs[0]


    def make_thunk(self, node, storage_map, compute_map, no_recycling):
        ### TODO: use inputs
        img_var, filter_var = node.inputs
        out_var, = node.outputs

        img_cell = storage_map[img_var]
        filter_cell = storage_map[filter_var]
        out_cell = storage_map[out_var]

        img_cnda = storage_map[img_var][0]
        filters_cnda = storage_map[filter_var][0]

        in_ = Input(*self.img_shp)
        fb_ = Filterbank(*self.ker_shp)
        out_ = Output(*self.out_shp)

        fop = FilterOp(in_, fb_, out_, ctxt=self.context, **self.fop_kwargs)

        def thunk():
            ival = img_cell[0]
            fval = filter_cell[0]
            # TODO: slice input image into 4s
            #       rather than copy it to memory and back
            in_[:] = np.asarray(ival[0])
            fb_[:] = np.asarray(fval)
            out_[:] = 0
            fop()
            outval = out_[:][None, :, :, :]
            if isinstance(out_var.type, theano.tensor.TensorType):
                out_cell[0] = outval
                print out_cell[0].shape, out_cell[0].sum()
            else:
                out_cell[0] = theano.sandbox.cuda.CudaNdarray(outval)
            compute_map[out_var][0] = 1

        thunk.inputs = [storage_map[n] for n in node.inputs]
        thunk.outputs = [storage_map[n] for n in node.outputs]
        thunk.lazy = False
        return thunk



def rand(shp, dtype):
    return np.random.rand(*shp).astype(dtype)


@with_cuda_context
def test_match_theano(context, dtype='float32', n_imgs=1, rows=3, cols=3, frows=2,
        fcols=2,
        channels=1, n_filters=4):

    print 'IMAGES SHAPE', n_imgs, rows, cols, channels
    print 'FILTERS SHAPE', n_filters, frows, fcols, channels

    # prefix A is for theano's ConvOp
    A_ishp = (n_imgs, channels, rows, cols)
    A_kshp = (n_filters, channels, frows, fcols)

    ival = rand(A_ishp, dtype)
    fval = rand(A_kshp, dtype)

    print 'IMGS'
    print ival[0].shape
    print ''
    print 'FILTERS'
    print fval[0].shape
    print ''
    #print convolve(ival[0].transpose(1, 2, 0), fval[0].transpose(1, 2, 0),
    #        mode='constant')[:-1,:-1,0]
    print '-' * 80

    A_imgs = theano.shared(ival)
    A_filters = theano.shared(fval)
    A_out = theano.shared(rand((2, 2, 2, 2), dtype))

    # prefix B is for GCG Conv
    B_ishp = (n_imgs, rows, cols, channels)
    B_kshp = (n_filters, frows, fcols, channels)

    B_imgs = theano.shared(ival.transpose(0, 2, 3, 1).copy())
    B_filters = theano.shared(fval.transpose(0, 2, 3, 1)[:,::-1,::-1,:].copy())
    B_out = theano.shared(rand((2, 2, 2, 2), dtype))

    A_op = theano.tensor.nnet.conv.ConvOp(
            imshp=(channels, rows, cols),
            kshp=(frows, fcols),
            bsize=n_imgs,
            nkern=n_filters)

    B_op = GCG_FBCorr(B_ishp, B_kshp,
            use_tex1dfetch=False,
            context=context)

    f = theano.function([], [], updates={
        A_out: A_op(A_imgs, A_filters),
        B_out: B_op(B_imgs, B_filters),
        })
    f()
    # put things into Channel-major format
    Aval = A_out.get_value()
    Bval = B_out.get_value().transpose(0, 3, 1, 2)
    sanity = np.dot(
            ival[0,:,:frows,:fcols].flatten(),
            fval[0,:,::-1,::-1].flatten())
    print 'SANITY', ival[0,0,0,0]
    print sanity
    print Aval[0,0,0,0]
    print Bval[0,0,0,0]
    print 'max abs diff', abs(Aval - Bval).max()
    print 'fraction correct', (abs(Aval - Bval) < 0.001).sum(), '/', Aval.size
    assert Aval.shape == Bval.shape
    assert np.allclose(Aval, Bval)


@nose.plugins.attrib.attr('slow')
def test_fuzz(dtype='float32'):

    rng = np.random.RandomState(234)

    for i in range(2):
        # XXX add support for this to GCG_FBCorr
        #n_imgs = rng.randint(16) + 1
        n_imgs = 1

        rows = rng.randint(512) + 3
        cols = rows
        frows = rng.randint(min(cols - 2, 17)) + 1
        fcols = frows

        # XXX: relax this constaint, raise upper bound
        channels = rng.randint(32) + 4
        channels -= channels % 4

        # XXX: relax this constaint, raise upper bound
        n_filters = rng.randint(32) + 4
        n_filters -= n_filters % 4

        test_match_theano(n_imgs=n_imgs, rows=rows, cols=cols, frows=frows,
                fcols=fcols, channels=channels, n_filters=n_filters)

