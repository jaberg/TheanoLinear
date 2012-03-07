#raw
#include <stdio.h>
#end raw

// -- Define a few helpful constant(s) and macro(s)
#define uint unsigned int

#if $IMUL_FAST
#define IMUL(a, b) __mul24(a, b)
#else
#define IMUL(a, b) a * b
#end if

#set xyzw = ['x','y','z','w']

#if $USE_TEX1DFETCH
// -- Declare a float4 1D texture
// that will be used to fetch input values
texture<float4, 1, cudaReadModeElementType> tex_float4;
#end if

// -- Define constant memory buffer for the (sub) filter values
__constant__ float constant						\
[$INPUT_D]								\
[$N_FILTER_ROWS]							\
[$FILTER_W]								\
[$N_OUTPUT4S]								\
[$N_FILTERS];

extern "C" {

#for nk in xrange($N_KERNELS)

  __global__
  void cudafilter_kernel_${nk}
  (
   float4 *input
#for o in xrange($N_OUTPUT4S)
   , float4 *output$o
#end for
   )
  {

    // -- Shared-memory buffer for the input tiles
    __shared__ float shared_in			\
      [$BLOCK_H]				\
      [$N_FILTER_ROWS]				\
      [$INPUT_D]				\
      [$INPUT_BLOCK_W + ${int($PAD_SHARED)}] ;

    // -- Input/Output "pointers"
    const uint in_idx =				   \
      IMUL(IMUL(blockIdx.y, $BLOCK_H), $INPUT_W) + \
      IMUL(IMUL($nk, $INPUT_W), $N_FILTER_ROWS) +  \
      IMUL(threadIdx.y, $INPUT_W) +		   \
      IMUL(blockIdx.x, $BLOCK_W) + threadIdx.x ;

    const uint out_idx =				\
      IMUL(IMUL(blockIdx.y, $BLOCK_H), $OUTPUT_W) +	\
      IMUL(threadIdx.y, $OUTPUT_W) +			\
      IMUL(blockIdx.x, $BLOCK_W) + threadIdx.x ;

#if $SPILL
    // Create a shared-memory buffer to spill a register value
    // into shared memory and thus reduce the register count.
    __shared__ uint s_out_idx[$BLOCK_H][$BLOCK_W + ${int($PAD_SHARED)}];
    s_out_idx[threadIdx.y][threadIdx.x] = out_idx;
#end if

    // -------------------------------------------------------------------------
    // -- Load input to shared-memory
    // -------------------------------------------------------------------------
#for nfr in xrange($N_FILTER_ROWS)
#for i in xrange($N_LOAD_ITERATIONS)
#if $i==($N_LOAD_ITERATIONS-1)
    if( (threadIdx.x + IMUL($BLOCK_W, $i)) < $INPUT_BLOCK_W )
#end if
      {

#if $USE_TEX1DFETCH
	float4 ival = tex1Dfetch(tex_float4, in_idx + IMUL($INPUT_W, $nfr) + IMUL($BLOCK_W, $i));
#else
	float4 ival = input[in_idx + IMUL($INPUT_W, $nfr) + IMUL($BLOCK_W, $i)];
#end if

#for d in xrange($N_FILTERS)
	shared_in[threadIdx.y][$nfr][$d][threadIdx.x + IMUL($BLOCK_W, $i)] = ival.$xyzw[$d];
#end for

      }
#end for
#end for

    __syncthreads();

    // -------------------------------------------------------------------------
    // -- Compute dot products (fully unrolled)
    // -------------------------------------------------------------------------
    float value, weight;

#for o in xrange($N_OUTPUT4S)
#for n in xrange($N_FILTERS)
    float sum${o}${n} = 0;
#end for
#end for

#for d in xrange($INPUT_D)
#for nfr in xrange($N_FILTER_ROWS)
#for i in xrange($FILTER_W)
    value = shared_in[threadIdx.y][$nfr][$d][threadIdx.x+$i];
#for o in xrange($N_OUTPUT4S)
#for n in xrange($N_FILTERS)
    weight = constant[$d][$nfr][$i][$o][$n];
    sum${o}${n} += value*weight;
#end for
#end for
#end for
#end for
#end for

    // -------------------------------------------------------------------------
    // -- Output results
    // -------------------------------------------------------------------------
#for o in xrange($N_OUTPUT4S)
#for n in xrange($N_FILTERS)
#if $SPILL
    output${o}[s_out_idx[threadIdx.y][threadIdx.x]].$xyzw[$n] += sum${o}${n};
#else
    output${o}[out_idx].$xyzw[$n] += sum${o}${n};
#end if
#end for
#end for

  }
#end for



}

