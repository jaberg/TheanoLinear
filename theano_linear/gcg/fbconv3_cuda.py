#!/usr/bin/env python
# -*- coding: utf-8 -*-

# TODO: refactor this old dude, pep8, etc.
# TODO: remove print, replace with log

import os
import numpy as np
import logging
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

# CUDA
from pycuda import driver, gpuarray, compiler, tools

#import os
from os import path

from Cheetah.Template import Template

from fbconv3_utils import InvalidConfig, MYPATH

# -----------------------------------------------------------------------------
PADFACTOR_H = 16
PADFACTOR_W = 16


# =============================================================================
class FilterOp(object):

    # -------------------------------------------------------------------------
    def __init__(self,
                 in_, fb_, out_,
                 # -- meta-programming parameters
                 block_w=8,
                 block_h=8,
                 n_filter_rows=1,  # 'all', 1, 2, 4, 8, 16
                 n_output4s='all',  # 'all', 1, 2, 4, 8, 16
                 spill=False,  # False, True
                 imul_fast=True,  # True, False
                 pad_shared=True,  # True, False
                 use_tex1dfetch=True,  # True, False
                 maxrregcount=None,  # None, 8, 16, 32
                 # --
                 use_fast_math=False,
                 ctxt=None,
                 ):
        assert ctxt is not None

        def get_device_attribute(name):
            return ctxt.get_device().get_attribute(
                    driver.device_attribute.__dict__[name])

        # block of threads
        block = block_w, block_h, 1

        max_block_w = get_device_attribute("MAX_BLOCK_DIM_X")
        if block_w > max_block_w:
            raise InvalidConfig("block_w (%d) is too large (max_block_w=%d)."
                                % (block_w, max_block_w))

        max_block_h = get_device_attribute("MAX_BLOCK_DIM_Y")
        if block_h > max_block_h:
            raise InvalidConfig("block_h (%d) is too large (max_block_h=%d)."
                                % (block_h, max_block_h))

        max_threads = get_device_attribute("MAX_THREADS_PER_BLOCK")
        if block_w * block_h > max_threads:
            raise InvalidConfig("Too many threads! "
                                "The current device supports %d threads, "
                                "but you asked %d (block_w=%d * block_h=%d)."
                                % (max_threads,
                                   block_w * block_h, block_w, block_h))

        # XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
        garr_out_l = out_._garr_l
        garr_in_l = in_._garr_l

        # original shapes
        in_h, in_w, in_d = in_.height, in_.width, in_.depth
        fb_n, fb_h, fb_w, fb_d = \
                fb_.n_filters, fb_.height, fb_.width, fb_.depth
        out_h, out_w, out_d = out_.height, out_.width, out_.depth

        assert out_d == fb_n

        # padded shapes
        garr_in_h, garr_in_w, garr_in_d = in_._garr_l[0].shape
        garr_out_h, garr_out_w, garr_out_d = out_._garr_l[0].shape

        if n_filter_rows == 'all':
            n_filter_rows = fb_h

        if n_output4s == 'all':
            n_output4s = len(garr_out_l)

        if fb_h % n_filter_rows != 0:
            raise InvalidConfig("fb_h (%d) "
                                "is not a multiple of n_filter_rows (%d)"
                                % (fb_h, n_filter_rows))

        if len(garr_out_l) % n_output4s != 0:
            raise InvalidConfig("len(garr_out_l) (%d) "
                                "is not a multiple of n_output4s (%d)"
                                % (len(garr_out_l), n_output4s))

        #self.in_ = in_
        #self.fb_ = fb_
        #self.out_ = out_

        # grid of blocks
        grid = (int(np.ceil(1.*garr_out_w/block_w)),
                (int(np.ceil(1.*garr_out_h/block_h))), 1)
        grid_w, grid_h = grid[:2]

        max_grid_w = get_device_attribute("MAX_GRID_DIM_X")
        if grid_w > max_grid_w:
            raise InvalidConfig("grid_w (%d) is too large (max_grid_w=%d)."
                                % (grid_w, max_grid_w))

        max_grid_h = get_device_attribute("MAX_GRID_DIM_Y")
        if grid_h > max_grid_h:
            raise InvalidConfig("grid_h (%d) is too large (max_grid_h=%d)."
                                % (grid_h, max_grid_h))

        # -- generate gpu code

        # - define template options
        topts = {}
        # filters
        topts['FILTER_H'] = fb_h
        topts['FILTER_W'] = fb_w
        # input
        topts['INPUT_H'] = garr_in_h
        topts['INPUT_W'] = garr_in_w
        topts['INPUT_D'] = garr_in_d
        # output
        topts['OUTPUT_H'] = garr_out_h
        topts['OUTPUT_W'] = garr_out_w
        # blocks
        topts['BLOCK_W'] = block_w
        topts['BLOCK_H'] = block_h
        topts['INPUT_BLOCK_W'] = (block_w+fb_w-1) if ((block_w+fb_w-1)<garr_in_w) else garr_in_w
        topts['N_LOAD_ITERATIONS'] = int(np.ceil((block_w+fb_w-1.)/block_w))

        topts['PAD_SHARED'] = pad_shared
        topts['SPILL'] = spill

        # XXX: review
        topts['USE_TEX1DFETCH'] = use_tex1dfetch
        topts['N_OUTPUT4S'] = n_output4s
        topts['N_FILTERS'] = 4

        # XXX: tempppp
        assert fb_h % n_filter_rows == 0
        topts['N_FILTER_ROWS'] = n_filter_rows
        n_kernels = fb_h / n_filter_rows
        topts['N_KERNELS'] = n_kernels

        # XXX: TEMP
        topts['IMUL_FAST'] = imul_fast

        # - generate source from template
        basename = path.join(MYPATH, path.splitext(__file__)[-2])
        tmpl_basename = path.join(basename)
        tmpl_fname = tmpl_basename + ".template.cu"
        tmpl = Template(file=tmpl_fname, searchList=[topts])
        outstr = tmpl.respond()

        # -- nvcc flags
        opt_l = os.environ.get("PYCUDA_DEFAULT_NVCC_FLAGS", "").split()
        if maxrregcount is not None:
            opt_l += ["--maxrregcount=%d" % maxrregcount]
        if use_fast_math:
            opt_l += ["--use_fast_math"]
        log.info("NVCC options: %s", opt_l)

        # -- compile source
        try:
            cubin_str = compiler.compile(outstr, options=opt_l)
        except driver.CompileError, err:
            # XXX: better handling of known errors
            if "#error -- unsupported GNU version" in err.stderr:
                log.critical(
                    "Please use the env variable "
                    "PYCUDA_DEFAULT_NVCC_FLAGS to specify a "
                    "compiler bindir < gcc-4.4, for example: "
                    "export PYCUDA_DEFAULT_NVCC_FLAGS="
                    "'--compiler-bindir=$(gcc-config "
                    "-B x86_64-pc-linux-gnu-4.4.5)'")
                raise err
            #log.critical("%s: %s", err.msg, err.stderr)
            raise err
            #else:
                #raise InvalidConfig("%s: %s" % (err.msg, err.stderr))

        # - XXX
        mod = driver.module_from_buffer(cubin_str)

        cudafunc_l = [mod.get_function('cudafilter_kernel_%d' % nk)
                      .prepare("P"*(n_output4s+1), block=block)
                      for nk in xrange(n_kernels)]

        # -- reference to texture memory
        if use_tex1dfetch:
            if fb_d == 1:
                tex = mod.get_texref("tex_float")
                tex.set_format(driver.array_format.FLOAT, 1)
            else:
                tex = mod.get_texref("tex_float4")
                tex.set_format(driver.array_format.FLOAT, 4)

        # -- reference to constant memory
        const = mod.get_global("constant")[0]

        # -- prepare function calls
        grid2 = grid[:2]

        def fill_const_nocache(j, iz, oz):

            fb_sub = fb_[oz*4*n_output4s : (oz+1)*4*n_output4s,
                         j*n_filter_rows : (j+1)*n_filter_rows,
                         :,
                         iz*4 : (iz+1)*4]

            fb_sub = np.swapaxes(fb_sub, 0, 3)

            fb_sub = np.ascontiguousarray(fb_sub)

            max_const = get_device_attribute('TOTAL_CONSTANT_MEMORY')
            if fb_sub.nbytes > max_const:
                raise InvalidConfig("fb_sub.nbytes (%d) is too large (max_const=%d)."
                                    % (fb_sub.nbytes, max_const))

            driver.memcpy_htod(const, fb_sub.data)

        # update fb_sub if necessary
        cudafunc_call_l = []

        max_regs = get_device_attribute('MAX_REGISTERS_PER_BLOCK')
        max_smem = get_device_attribute('MAX_SHARED_MEMORY_PER_BLOCK')

        for o4z in xrange(len(garr_out_l)/n_output4s):

            for iz, garr_in in enumerate(garr_in_l):

                if use_tex1dfetch:
                    # add call: bind input texture
                    cudafunc_call_l += [(garr_in.bind_to_texref, (tex,))]

                for j in xrange(fb_h/n_filter_rows):

                    # get cuda kernel function
                    cudafunc = cudafunc_l[j]

                    # make sure that the function can be run on the device
                    if cudafunc.num_regs > max_regs:
                        raise InvalidConfig(
                            "cudafunc.num_regs (%d) "
                            "is too large (max_regs=%d)."
                            % (fb_sub.cudafunc.num_regs, max_regs))

                    if cudafunc.shared_size_bytes > max_smem:
                        raise InvalidConfig(
                            "cudafunc.cudafunc.shared_size_bytes (%d) "
                            "is too large (max_regs=%d)."
                            % (fb_sub.cudafunc.num_regs, max_regs))

                    # add call: fill constant memory
                    cudafunc_call_l += [(fill_const_nocache, (j, iz, o4z))]

                    # add call: compute
                    cudafunc_call_l += [(cudafunc.prepared_call,
                                         [grid2, garr_in.gpudata] +
                                         [garr_out_l[o4z*n_output4s + i].gpudata
                                          for i in xrange(n_output4s)]
                                         )]

                # -
            # -
        # -

        self._cudafunc_call_l = cudafunc_call_l

    # -------------------------------------------------------------------------
    def __call__(self, **plugin_args):

        start = driver.Event()
        end = driver.Event()

        start.record()
        try:
            [func(*args) for func, args in self._cudafunc_call_l]
        except driver.LaunchError, err:
            raise InvalidConfig(err)

        # XXX: timing here?
        end.record()

        end.synchronize()

        return end.time_since(start)*1e-3


# =============================================================================
class Input(object):

    # -------------------------------------------------------------------------
    def __init__(self, height, width, depth, dtype='float32'):

        # constraints
        #assert nimgs == 1
        assert height == width
        assert depth % 4 == 0 or depth == 1
        assert dtype == 'float32'

        #self.nimgs = nimgs
        self.height = height
        self.width = width
        self.depth = depth
        self.dtype = dtype

        # padding
        padh = int(np.ceil(1.*height/PADFACTOR_H))*PADFACTOR_H
        padw = int(np.ceil(1.*width/PADFACTOR_W))*PADFACTOR_W
        padd = depth
        self._padded_shape = padh, padw, padd

        # gpuarray alloc
        try:
            if depth == 1:
                self._garr_l = [gpuarray.GPUArray((padh,padw,1),'float32')]
                ngarrs = 1
            else:
                ngarrs = int(np.ceil(depth / 4.))
                self._garr_l = [gpuarray.GPUArray((padh,padw,4),'float32') for _ in xrange(ngarrs)]
        except driver.MemoryError, err:
            raise InvalidConfig(err)

        if 0: # DEBUG XXX
            self._garr_tmp = driver.pagelocked_empty(self._garr_l[0].shape, self.dtype)
        else:
            self._garr_tmp = np.empty(self._garr_l[0].shape, self.dtype)
        self._arr_tmp = np.empty((ngarrs,)+self._padded_shape[:2]+(self._garr_l[0].shape[-1],), dtype='float32')

    # -------------------------------------------------------------------------
    def __getitem__(self, index):

        g_l = self._garr_l

        my_h, my_w, my_d = self.height, self.width, self.depth
        data = self._garr_tmp

        # --
        if my_d == 1 or my_d == 4:
            g_l[0].get(data)
            return data[:my_h,:my_w,:my_d][index].copy()

        # --
        ngarrs = len(g_l)

        harr = self._arr_tmp
        for i in xrange(ngarrs):
            g_l[i].get(data)
            harr[i] = data

        out = harr[:,:my_h,:my_w,:]
        out = out.reshape(ngarrs, my_h, my_w, 1, self._garr_l[0].shape[-1])
        out = out.swapaxes(0,3)
        out = out.reshape((my_h, my_w, my_d))

        if index == slice(None, None, None):
            return out.copy()
        else:
            return out[index].copy()

    # -------------------------------------------------------------------------
    def setitem_opt_tmp(self, value): # pragma: no cover
        g_l = self._garr_l

        for i in xrange(len(g_l)):
            g_l[i].set(value[:,:,i*4:(i+1)*4])

    # -------------------------------------------------------------------------
    def __setitem__(self, index, value):
        g_l = self._garr_l

        # -- clear data with unique value ?
        if (np.array(index, dtype=object)==slice(None,None,None)).all() \
               and np.array(value).size == 1:
            for i in xrange(len(g_l)):
                g_l[i].fill(float(value))

        # -- full update
        elif index == slice(None,None,None) \
                 and value.shape == self._padded_shape:
            for i in xrange(len(g_l)):
                g_l[i].set(np.ascontiguousarray(value[:,:,i*4:(i+1)*4]))

        # -- standard update
        else:
            if index != slice(None,None,None):
                harr = self[:]
                harr[index] = value
            else:
                harr = value
            tmp = driver.pagelocked_empty(g_l[0].shape, 'float32')
            h, w = self.height, self.width
            for i in xrange(len(g_l)):
                tmp[:h,:w,:4] = harr[:,:,i*4:(i+1)*4]
                g_l[i].set(tmp)

# =============================================================================
Output = Input

# =============================================================================
class Filterbank(object):

    # -------------------------------------------------------------------------
    def __init__(self, n_filters, height, width, depth, dtype='float32'):

        # constraints
        assert n_filters == 1 or n_filters % 4 == 0
        assert height == width
        assert depth == 1 or depth % 4 == 0

        assert dtype == 'float32' # for now

        self.n_filters = n_filters
        self.height = height
        self.width = width
        self.depth = depth
        self.dtype = dtype

        self._ndarray = np.ndarray((n_filters,height,width,depth), dtype=dtype)

    # -------------------------------------------------------------------------
    def __getitem__(self, key):
        return self._ndarray[key].copy()

    # -------------------------------------------------------------------------
    def __setitem__(self, key, value):

        shape = self._ndarray[key].shape
        value = np.array(value)

        if value.size > 1:
            self._ndarray[key] = value.reshape(shape).astype(self.dtype)
        else:
            self._ndarray[key] = value.astype(self.dtype)


