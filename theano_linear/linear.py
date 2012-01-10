"""
XXX
"""
from PIL import Image
import sys
import copy
import numpy
import theano
import theano.tensor as tensor
from theano.tensor.nnet.conv import conv2d, ConvOp
import pylearn.io.image_tiling

prod = numpy.prod

_ndarray_status_fmt='%(msg)s shape=%(shape)s min=%(min)f max=%(max)f'


def ndarray_status(x, fmt=_ndarray_status_fmt, msg="", **kwargs):
    kwargs.update(dict(
            msg=msg,
            min=x.min(),
            max=x.max(),
            mean=x.mean(),
            var = x.var(),
            shape=x.shape))
    return fmt%kwargs


class LinearTransform(object):
    def __init__(self, params):
        self.set_params(params)
    def set_params(self, params):
        self._params = list(params)
    def params(self):
        return list(self._params)
    def __add__(self, other):
        return LSum([self, other])
    def lmul(self, x, T=False):
        """mul(x, A) or mul(x, A.T)

        If T is True this method returns mul(x, A.T).
        If T is False this method returns mul(x, A).

        """
        return self._lmul(x, T)
    def lmul_outshape(self, xshp):
        xshp = tuple(xshp)
        examples = xshp[0:1]
        rest = xshp[1:]
        if tuple(rest) != self.col_shape():
            raise ValueError('structure mismatch. passed=%s, required=%s'%(rest,
                self.col_shape()))
        return examples + self.row_shape()
    def lmul_inshape(self, yshp):
        yshp = tuple(yshp)
        examples = yshp[0:1]
        rest = yshp[1:]
        if rest != self.row_shape():
            raise ValueError('structure mismatch. passed=%s, required=%s'%(rest,
                self.row_shape()))
        return examples + self.col_shape()
    def row_shape(self):
        return self._row_shape()
    def col_shape(self):
        return self._col_shape()

    def transpose(self, order=(1,0)):
        if tuple(order) != (1,0):
            raise ValueError('only matrix transposition is supported')
        return TransposeTransform(self)
    T = property(lambda self: self.transpose())

    def tile_columns(self, **kwargs):
        return self._tile_columns(**kwargs)

    def __str__(self):
        return self.__class__.__name__ +'{}'

    # SUBCLASSES OVER-RIDE THESE
    def _lmul(self, x, T):
        raise NotImplementedError('override me')
    def _row_shape(self):
        raise NotImplementedError('override me')
    def _col_shape(self):
        raise NotImplementedError('override me')
    def _tile_columns(self):
        raise NotImplementedError('override me')


class TransposeTransform(LinearTransform):
    def __init__(self, base):
        super(TransposeTransform, self).__init__([])
        self.base = base
    def transpose(self):
        return self.base
    def params(self):
        return self.base.params()
    def _lmul(self, x, T):
        return self.base._lmul(x, not T)
    def _row_shape(self):
        return self.base._col_shape()
    def _col_shape(self):
        return self.base._row_shape()
    def print_status(self):
        return self.base.print_status()
    def _tile_columns(self):
        # yes, it would be nice to do rows, but since this is a visualization
        # and there *is* no tile_rows, we fall back on this.
        return self.base._tile_columns()


class LConcat(LinearTransform):
    """
    Form a linear map of the form [A B ... Z].

    For this to be valid, A,B...Z must have identical row_shape.

    The col_shape defaults to being the concatenation of flattened output from
    each of A,B,...Z, but a col_shape tuple specified via the constructor will
    reshape that vector.
    """
    def __init__(self, Wlist, col_shape=None):
        super(LConcat, self).__init__([])
        self._Wlist = list(Wlist)
        if not isinstance(col_shape, (int,tuple,type(None))):
            raise TypeError('col_shape must be int or int tuple')
        self._col_sizes = [prod(w.col_shape()) for w in Wlist]
        if col_shape is None:
            self.__col_shape = sum(self._col_sizes),
        elif isinstance(col_shape, int):
            self.__col_shape = col_shape,
        else:
            self.__col_shape = tuple(col_shape)
        assert prod(self.__col_shape) == sum(self._col_sizes)
        self.__row_shape = Wlist[0].row_shape()
        for W in Wlist[1:]:
            if W.row_shape() != self.row_shape():
                raise ValueError('Transforms has different row_shape',
                        W.row_shape())

    def params(self):
        rval = []
        for W in self._Wlist:
            rval.extend(W.params())
        return rval
    def _lmul(self, x, T):
        if T:
            if len(self.col_shape())>1:
                x2 = x.flatten(2)
            else:
                x2 = x
            n_rows = x2.shape[0]
            offset = 0
            xWlist = []
            assert len(self._col_sizes) == len(self._Wlist)
            for size, W in zip(self._col_sizes, self._Wlist):
                # split the output rows into pieces
                x_s = x2[:,offset:offset+size]
                # multiply each piece by one transform
                xWlist.append(
                        W.lmul(
                            x_s.reshape(
                                (n_rows,)+W.col_shape()),
                            T))
                offset += size
            # sum the results
            rval = tensor.add(*xWlist)
        else:
            # multiply the input by each transform
            xWlist = [W.lmul(x,T).flatten(2) for W in self._Wlist]
            # join the resuls
            rval = tensor.join(1, *xWlist)
        return rval
    def _col_shape(self):
        return self.__col_shape
    def _row_shape(self):
        return self.__row_shape
    def _tile_columns(self):
        # hard-coded to produce RGB images
        arrays = [W._tile_columns() for W in self._Wlist]
        o_rows = sum([a.shape[0]+10 for a in arrays]) - 10
        o_cols = max([a.shape[1] for a in arrays])
        rval = numpy.zeros(
                (o_rows, o_cols, 3),
                dtype=arrays[0].dtype)
        offset = 0
        for a in arrays:
            if a.ndim==2:
                a = a[:,:,None] #make greyscale broadcast over colors
            rval[offset:offset+a.shape[0], 0:a.shape[1],:] = a
            offset += a.shape[0] + 10
        return rval
    def print_status(self):
        for W in self._Wlist:
            W.print_status()


class LSum(LinearTransform):
    def __init__(self, terms):
        self.terms = terms
        for t in terms[1:]:
            assert t.row_shape() == terms[0].row_shape()
            assert t.col_shape() == terms[0].col_shape()
    def params(self):
        rval = []
        for t in self.terms:
            rval.extend(t.params())
        return rval
    def _lmul(self, x, T):
        results = [t._lmul(x, T)]
        return tensor.add(*results)
    def _row_shape(self):
        return self.terms[0].col_shape()
    def _col_shape(self):
        return self.terms[0].row_shape()
    def print_status(self):
        for t in terms:
            t.print_status()
    def _tile_columns(self):
        raise NotImplementedError('TODO')


class MatrixMul(LinearTransform):
    # Works for Sparse and TensorType matrices
    def __init__(self, W, row_shape=None, col_shape=None):
        """

        If W is not shared variable, row_shape and col_shape must be
        specified.
        """
        super(MatrixMul, self).__init__([W])
        self._W = W
        Wval = None
        if row_shape is None:
            Wval = W.get_value(borrow=True)
            rows, cols = Wval.shape
            self.__row_shape = rows,
        else:
            self.__row_shape = tuple(row_shape)
        if col_shape is None:
            if Wval is None:
                Wval = W.get_value(borrow=True)
                rows, cols = Wval.shape
            self.__col_shape = cols,
        else:
            self.__col_shape = tuple(col_shape)

    def _lmul(self, x, T):
        if T:
            W = self._W.T
            rshp = tensor.stack(x.shape[0], *self.__row_shape)
        else:
            W = self._W
            rshp = tensor.stack(x.shape[0], *self.__col_shape)
        rval = theano.dot(x.flatten(2), W).reshape(rshp)
        return rval
    def _row_shape(self):
        return self.__row_shape
    def _col_shape(self):
        return self.__col_shape

    def print_status(self):
        print ndarray_status(self._W.get_value(borrow=True), msg=self._W.name)
    def _tile_columns(self, channel_major=False, scale_each=False,
            min_dynamic_range=1e-4, **kwargs):
        W = self._W.get_value(borrow=False).T
        shape = self.row_shape()
        if channel_major:
            W.shape = (W.shape[0:1]+shape)
            W = W.transpose(0,2,3,1) #put colour last
        else:
            raise NotImplementedError()
        return pylearn.io.image_tiling.tile_slices_to_image(W,
                scale_each=scale_each,
                **kwargs)


def tile_conv_weights(w,flip=False, scale_each=False):
    """
    Return something that can be rendered as an image to visualize the filters.
    """
    if w.shape[1] != 3:
        raise NotImplementedError('not rgb', w.shape)
    if w.shape[2] != w.shape[3]:
        raise NotImplementedError('not square', w.shape)
    wmin, wmax = w.min(), w.max()
    if not scale_each:
        w = numpy.asarray(255 * (w - wmin) / (wmax - wmin + 1e-6), dtype='uint8')
    trows, tcols= pylearn.io.image_tiling.most_square_shape(w.shape[0])
    outrows = trows * w.shape[2] + trows-1
    outcols = tcols * w.shape[3] + tcols-1
    out = numpy.zeros((outrows, outcols,3), dtype='uint8')

    tr_stride= 1+w.shape[1]
    for tr in range(trows):
        for tc in range(tcols):
            # this is supposed to flip the filters back into the image
            # coordinates as well as put the channels in the right place, but I
            # don't know if it really does that
            tmp = w[tr*tcols+tc].transpose(1,2,0)[
                             ::-1 if flip else 1,
                             ::-1 if flip else 1]
            if scale_each:
                tmp = numpy.asarray(255*(tmp - tmp.min()) / (tmp.max() - tmp.min() + 1e-6),
                        dtype='uint8')
            out[tr*(1+w.shape[2]):tr*(1+w.shape[2])+w.shape[2],
                    tc*(1+w.shape[3]):tc*(1+w.shape[3])+w.shape[3]] = tmp
    return out


class LConv(LinearTransform):
    """
    XXX
    """

    def __init__(self, filters, img_shape, subsample=(1,1), border_mode='valid',
            filters_shape=None, message=""):
        super(LConv, self).__init__([filters])
        self._filters = filters
        if filters_shape is None:
            self._filters_shape = tuple(filters.get_value().shape)
        else:
            self._filters_shape = tuple(filters_shape)
        self._img_shape = tuple(img_shape)
        self._subsample = tuple(subsample)
        self._border_mode = border_mode
        if message:
            self._message = message
        else:
            self._message = filters.name
        if not len(self._img_shape)==4:
            raise TypeError('need 4-tuple shape', self._img_shape)
        if not len(self._filters_shape)==4:
            raise TypeError('need 4-tuple shape', self._filters_shape)

    def _lmul(self, x, T):
        if T:
            dummy_v = tensor.tensor4()
            z_hs = conv2d(dummy_v, self._filters,
                    image_shape=self._img_shape,
                    filter_shape=self._filters_shape,
                    subsample=self._subsample,
                    border_mode=self._border_mode,
                    )
            xfilters, xdummy = z_hs.owner.op.grad((dummy_v, self._filters), (x,))
            return xfilters
        else:
            return conv2d(
                    x, self._filters,
                    image_shape=self._img_shape,
                    filter_shape=self._filters_shape,
                    subsample=self._subsample,
                    border_mode=self._border_mode,
                    )

    def _row_shape(self):
        return self._img_shape[1:]

    def _col_shape(self):
        rows_cols = ConvOp.getOutputShape(
                self._img_shape[2:],
                self._filters_shape[2:],
                self._subsample,
                self._border_mode)
        rval = (self._filters_shape[0],)+tuple(rows_cols)
        return rval

    def _tile_columns(self, scale_each=True, **kwargs):
        return pylearn.io.image_tiling.tile_slices_to_image(
                self._filters.get_value()[:,:,::-1,::-1].transpose(0,2,3,1),
                scale_each=scale_each,
                **kwargs)

    def print_status(self):
        print ndarray_status(
                self._filters.get_value(borrow=True),
                msg='LConv{%s}'%self._message)

if 0:
    class LinearProd(LinearTransform):
        """ For linear transformations [A,B,C]
        this represents the linear transformation A(B(C(x))).
        """
        def __init__(self, linear_transformations):
            self._linear_transformations = linear_transformations
        def dot(self, x):
            return reduce(
                    lambda t,a:t.dot(a),
                    self._linear_transformations,
                    x)
        def transpose_dot(self, x):
            return reduce(
                    lambda t, a: t.transpose_dot(a),
                    reversed(self._linear_transformations),
                    x)
        def params(self):
            return reduce(
                    lambda t, a: a + t.params(),
                    self._linear_transformations,
                    [])

