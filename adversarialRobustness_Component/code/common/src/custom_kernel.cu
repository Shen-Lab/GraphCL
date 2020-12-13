
#include <stdio.h>
#include <math.h>
#include <float.h>

#include "custom_kernel.h"

struct SharedMem
{
  __device__ double *getPointer() {
    extern __shared__ double s_double[];
    return s_double;
  }
};


struct MaxFloat
{
  __device__ __forceinline__ double operator()(double max, float v) const {
    return max > static_cast<double>(v) ? max : static_cast<double>(v);
  }
};

struct Max
{
  __device__ __forceinline__ double operator()(double x, double y) const {
    return x > y ? x : y;
  }
};

struct Add
{
  __device__ __forceinline__ double operator()(double x, double y) const {
    return x + y;
  }
};

struct AddFloat
{
  __device__ __forceinline__ double operator()(double sum, float v) const {
    return sum + v;
  }
};

struct SumExpFloat
{
  __device__ __forceinline__ SumExpFloat(double v)
    : max_k(v) {}

  __device__ __forceinline__ double operator()(double sum, float v) const {
    return sum + static_cast<double>(exp((double)v - max_k));
  }

  const double max_k;
};

template <typename Reduction>
__device__ __forceinline__ double
blockReduce(double* smem, double val,
            const Reduction& r,
            double defaultVal)
{
  // To avoid RaW races from chaining blockReduce calls together, we need a sync here
  __syncthreads();

  smem[threadIdx.x] = val;

  __syncthreads();

  double warpVal = defaultVal;

  // First warp will perform per-warp reductions for the remaining warps
  if (threadIdx.x < 32) {
    int lane = threadIdx.x % 32;
    if (lane < blockDim.x / 32) {
#pragma unroll
      for (int i = 0; i < 32; ++i) {
        warpVal = r(warpVal, smem[lane * 32 + i]);
      }
      smem[lane] = warpVal;
    }
  }

  __syncthreads();

  // First thread will perform a reduction of the above per-warp reductions
  double blockVal = defaultVal;

  if (threadIdx.x == 0) {
    for (int i = 0; i < blockDim.x / 32; ++i) {
      blockVal = r(blockVal, smem[i]);
    }
    smem[0] = blockVal;
  }

  // Sync and broadcast
  __syncthreads();
  return smem[0];
}


template <typename Reduction, int ILP>
__device__ __forceinline__ double
ilpReduce(float* data,
          int size,
          const Reduction& r,
          double defaultVal)
{
  double threadVal = defaultVal;
  int offset = threadIdx.x;

  int last = size % (ILP * blockDim.x);

  // Body (unroll by ILP times)
  for (; offset < size - last; offset += blockDim.x * ILP) {
    float tmp[ILP];

#pragma unroll
    for (int j = 0; j < ILP; ++j)
      tmp[j] = data[offset + j * blockDim.x];

#pragma unroll
    for (int j = 0; j < ILP; ++j)
      threadVal = r(threadVal, tmp[j]);
  }

  // Epilogue
  for (; offset < size; offset += blockDim.x)
    threadVal = r(threadVal, data[offset]);

  return threadVal;
}

template <int ILP>
__global__ void cunn_SoftMaxForward(float *output, float *input, long* ps)
{
  SharedMem smem;
  double *buffer = smem.getPointer();
  // forward pointers to batch[blockIdx.x]
  // each block handles a sample in the mini-batch
  long ofs = (blockIdx.x == 0) ? 0 : ps[blockIdx.x - 1];
  long n_ele = ps[blockIdx.x] - ofs;
  input += ofs;
  output += ofs;

  // find the max
  double threadMax = ilpReduce<MaxFloat, ILP>(input, n_ele, MaxFloat(), -DBL_MAX);

  double max_k = blockReduce<Max>(buffer, threadMax, Max(), -DBL_MAX);
  // float max_k_non_accum = static_cast<float>(max_k);

  // reduce all values
  double threadExp = ilpReduce<SumExpFloat, ILP>(input, n_ele, SumExpFloat(max_k), static_cast<double>(0));

  double sumAll = blockReduce<Add>(buffer, threadExp, Add(), static_cast<double>(0));

//   Epilogue<T, double> epilogue(max_k_non_accum, sumAll);
  // float logsum = max_k_non_accum + static_cast<float>(log(sumAll));
  double logsum = max_k + log(sumAll);

  int offset = threadIdx.x;
  int last = n_ele % (ILP * blockDim.x);
  for (; offset < n_ele - last; offset += blockDim.x * ILP) {
    float tmp[ILP];

#pragma unroll
    for (int j = 0; j < ILP; ++j)
      tmp[j] = input[offset + j * blockDim.x];

#pragma unroll
    for (int j = 0; j < ILP; ++j)
      output[offset + j * blockDim.x] = (double)tmp[j] - logsum;
  }

  for (; offset < n_ele; offset += blockDim.x)
    output[offset] = (double)input[offset] - logsum;
}

template <int ILP>
__global__ void cunn_SoftMaxBackward(float *gradInput, float *output, float *gradOutput, long* ps)
{
  SharedMem smem;  
  double *buffer = smem.getPointer();
  long ofs = (blockIdx.x == 0) ? 0 : ps[blockIdx.x - 1];
  long n_ele = ps[blockIdx.x] - ofs;

  gradInput += ofs;
  output += ofs;
  gradOutput += ofs;

  double threadSum = ilpReduce<AddFloat, 4>(
      gradOutput, n_ele, AddFloat(), double(0));
  double sum_k = blockReduce<Add>(
        buffer, threadSum, Add(), double(0));

  int offset = threadIdx.x;
  int last = n_ele % (ILP * blockDim.x);
  for (; offset < n_ele - last; offset += blockDim.x * ILP) {
    float tmpGradOutput[ILP];
    float tmpOutput[ILP];

#pragma unroll
    for (int j = 0; j < ILP; ++j) {
      tmpGradOutput[j] = gradOutput[offset + j * blockDim.x];
      tmpOutput[j] = output[offset + j * blockDim.x];
    }

#pragma unroll
    for (int j = 0; j < ILP; ++j)
      gradInput[offset + j * blockDim.x] = tmpGradOutput[j] - exp((double)tmpOutput[j]) * sum_k;
  }

  for (; offset < n_ele; offset += blockDim.x)
    gradInput[offset] = gradOutput[offset] - exp((double)output[offset]) * sum_k;
}


void HostSoftMaxForward(cudaStream_t stream, float *input, float *output, long* ps, int bsize)
{
// This kernel spawns a block of 1024 threads per each element in the batch.
// XXX: it assumes that inner_size == 1

  dim3 grid(bsize);
  dim3 block(1024);

  cunn_SoftMaxForward<2>
  <<<grid, block, block.x * sizeof(double), stream>>>(
    output, input, ps
  );  

  // THCudaCheck(cudaGetLastError());
}

void HostSoftMaxBackward(cudaStream_t stream, float *gradOutput, float *gradInput, float *output, long* ps, int bsize)
{
  dim3 grid(bsize);
  dim3 block(1024);

  cunn_SoftMaxBackward<2>
  <<<grid, block, block.x * sizeof(double), stream>>>(
        gradInput, output, gradOutput, ps
    );
  
  // THCudaCheck(cudaGetLastError());
}

__global__ void JaggedArgmaxKernel(long* dst, float *orig_ptr, long* ps)
{
    __shared__ long buffer[256];

    long ofs = (blockIdx.x == 0) ? 0 : ps[blockIdx.x - 1];
    long cols = ps[blockIdx.x] - ofs;

    float* row_ptr = orig_ptr + ofs;

    int i_start = threadIdx.x;
    int i_end = cols;
    int i_step = blockDim.x;
    if (i_start < cols)
      buffer[threadIdx.x] = i_start;
    for (int i = i_start + i_step; i < i_end; i += i_step)
    {
      if (row_ptr[i] > row_ptr[buffer[threadIdx.x]])
        buffer[threadIdx.x] = i;
    }
    __syncthreads();

    int shift;
    for (int i = 8 - 1; i >= 0; --i)
    {
    	shift = 1 << i;
    	if (threadIdx.x < shift && threadIdx.x + shift < cols)
    	{
        if (row_ptr[buffer[threadIdx.x + shift]] > row_ptr[buffer[threadIdx.x]])
          buffer[threadIdx.x] = buffer[threadIdx.x + shift];
    	}
		  __syncthreads();
    }
    if (threadIdx.x == 0)
    	dst[blockIdx.x] = buffer[0];
}

void HostArgmaxForward(cudaStream_t stream, float *input, long *output, long* ps, int bsize)
{
  dim3 grid(bsize);
  dim3 block(256);  

  JaggedArgmaxKernel<<<grid, block, 0, stream>>>(output, input, ps);
}

__global__ void JaggedMaxKernel(float* vmax, long* idxes, float *orig_ptr, long* ps)
{
    __shared__ long buffer[256];  
    long ofs = (blockIdx.x == 0) ? 0 : ps[blockIdx.x - 1];
    long cols = ps[blockIdx.x] - ofs;

    float* row_ptr = orig_ptr + ofs;

    int i_start = threadIdx.x;
    int i_end = cols;
    int i_step = blockDim.x;
    if (i_start < cols)
      buffer[threadIdx.x] = i_start;
    for (int i = i_start + i_step; i < i_end; i += i_step)
    {
      if (row_ptr[i] > row_ptr[buffer[threadIdx.x]])
        buffer[threadIdx.x] = i;
    }
    __syncthreads();

    int shift;
    for (int i = 8 - 1; i >= 0; --i)
    {
    	shift = 1 << i;
    	if (threadIdx.x < shift && threadIdx.x + shift < cols)
    	{
        if (row_ptr[buffer[threadIdx.x + shift]] > row_ptr[buffer[threadIdx.x]])
          buffer[threadIdx.x] = buffer[threadIdx.x + shift];
    	}
		  __syncthreads();
    }
    if (threadIdx.x == 0)
    {
      idxes[blockIdx.x] = buffer[0];
      vmax[blockIdx.x] = row_ptr[buffer[0]];
    }
}

void HostMaxForward(cudaStream_t stream, float *input, float* vmax, long *idxes, long* ps, int bsize)
{
  dim3 grid(bsize);
  dim3 block(256);  

  JaggedMaxKernel<<<grid, block, 0, stream>>>(vmax, idxes, input, ps); 
}

#define min(x, y) (x < y ? x : y)

__global__ void GLapNormKernel(long* row_indices, long* col_indices, float* p_v, float* p_norm, int nnz)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    
    if(i < nnz)
    {
      float norm = p_norm[ row_indices[i] ] * p_norm[ col_indices[i] ];
      p_v[i] /= norm;
    }
}

void HostGLapNorm(cudaStream_t stream, long* row_indices, long* col_indices, float* p_v, float* p_norm, int nnz)
{ 
  int thread_num = min(1024, nnz);
  int blocksPerGrid = (nnz + thread_num - 1) / thread_num;
  
  GLapNormKernel<<<blocksPerGrid, thread_num, 0, stream>>> (row_indices, col_indices, p_v, p_norm, nnz);
}

__global__ void GDegreeNormKernel(long* row_indices, float* p_v, float* p_norm, int nnz)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    
    if(i < nnz)
    {
      float norm = p_norm[ row_indices[i] ];
      p_v[i] /= norm;
    }
}

void HostGDegreeNorm(cudaStream_t stream, long* row_indices, float* p_v, float* p_norm, int nnz)
{ 
  int thread_num = min(1024, nnz);
  int blocksPerGrid = (nnz + thread_num - 1) / thread_num;
  
  GDegreeNormKernel<<<blocksPerGrid, thread_num, 0, stream>>> (row_indices, p_v, p_norm, nnz);
}