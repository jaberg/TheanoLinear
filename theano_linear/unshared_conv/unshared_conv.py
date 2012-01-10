import numpy
import theano

from ..linear import LinearTransform


class ImageLocalDot(LinearTransform):
    """
    ImageLocalDot is an linear operation computationlly similar to
    convolution in the spatial domain, except that whereas convolution
    applying a single filter or set of filters across an image, the
    ImageLocalDot has different filterbanks for different points in the image.

    Mathematically, this is a general linear transform except for a
    restriction that filters are 0 outside of a spatially localized patch
    within the image.

    """

    def __init__(self, filters, img_shape,
            subsample=(1, 1),
            border_mode='valid',
            filters_shape=None,
            message=""):
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
            # left-multiply transpose of self with x
            raise NotImplementedError()
        else:
            # left-multiply self with x
            raise NotImplementedError()

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


class FilterActs(theano.Op):
    """
    Images of shape: colors x 
    Filters are of shape:
        channels
    """
    def __init__(self,
            module_stride=1,
            ):
        self.module_stride = module_stride

    def _attributes(self):
        return (
                self.module_stride,
                )

    def __eq__(self, other):
        return (type(self) == type(other)
                and self._attributes() == other._attributes())

    def __hash__(self):
        return hash((type(self), self._attributes()))

    def make_node(self, images, filters):
        return theano.gof.Apply(self,
                [images, filters],
                [images.type()])

    def perform(self, node, iargs, ostor):
        images, filters = iargs

        icolors, iheight, iwidth, icount = images.shape
        fgroups_h, fgroups_w, fcolors, fheight, fwidth, fcount = filters.shape
        # groups are "modules" in Alex's code

        if iheight != iwidth:
            raise ValueError("non-square image argument",
                    (iheight, iwidth))
        if fheight != fwidth:
            raise ValueError("non-square filter shape",
                    (fheight, fwidth))
        if fgroups_h != fgroups_w:
            raise ValueError('non-square filter grouping',
                    (fgroups_h, fgroups_w))
        if icolors != fcolors:
            raise ValueError("color counts don't match",
                    (icolors, fcolors))

        target = numpy.zeros((fcount, fgroups_h, fgroups_w, icount),
                dtype=images.dtype)

        for filter_idx in xrange(fcount):
            for ogroup_h_idx in xrange(fgroups_h):
                for ogroup_w_idx in xrange(fgroups_w):
                    img_h_offset = ogroup_h_idx * self.module_stride
                    img_w_offset = ogroup_w_idx * self.module_stride
                    ipatches = images[:,
                            img_h_offset:img_h_offset + fheight,
                            img_w_offset:img_w_offset + fwidth,
                            :].reshape(-1, icount)
                    fpatch = filters[ogroup_h_idx,
                            ogroup_w_idx,
                            :,
                            :,
                            :,
                            filter_idx].flatten()
                    target[filter_idx,
                            ogroup_h_idx,
                            ogroup_w_idx,
                            :] = (ipatches.T * fpatch).sum(axis=1)
        ostor[0][0] = target


    def unready_c_code(self, node, nodename, inames, onames, sub):
        images, filters = inames
        targets, = onames
        fail = sub['fail']
        return """
        """ % locals()
