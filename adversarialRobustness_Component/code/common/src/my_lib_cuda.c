#include <THC/THC.h>

#include "custom_kernel.h"

// this symbol will be resolved automatically from PyTorch libs
extern THCState *state;

int jagged_log_softmax_forward_cuda(THCudaTensor *logits, THCudaLongTensor *prefix_sum, THCudaTensor *output)
{
  logits = THCudaTensor_newContiguous(state, logits);
  THCudaTensor_resizeAs(state, output, logits);
  
  float *input_data_base  = THCudaTensor_data(state, logits);
  long* ps = THCudaLongTensor_data(state, prefix_sum);
  float *output_data_base  = THCudaTensor_data(state, output);
  
  int bsize = (int)prefix_sum->size[0];
  cudaStream_t stream = THCState_getCurrentStream(state);
  HostSoftMaxForward(stream, input_data_base, output_data_base, ps, bsize); 

  THCudaTensor_free(state, logits);
  return 1;
}

int jagged_log_softmax_backward_cuda(THCudaTensor *output, THCudaTensor *grad_output, THCudaLongTensor *prefix_sum, THCudaTensor *grad_input)
{
  output = THCudaTensor_newContiguous(state, output);
  grad_output = THCudaTensor_newContiguous(state, grad_output);

  THCudaTensor_resizeAs(state, grad_input, grad_output);
  float *output_data_base  = THCudaTensor_data(state, output);  
  float *gradOutput_data_base  = THCudaTensor_data(state, grad_output);
  long* ps = THCudaLongTensor_data(state, prefix_sum);
  float *gradInput_data_base  = THCudaTensor_data(state, grad_input);
  
  int bsize = (int)prefix_sum->size[0];
  cudaStream_t stream = THCState_getCurrentStream(state);
  HostSoftMaxBackward(stream, gradOutput_data_base, gradInput_data_base, output_data_base, ps, bsize); 
  THCudaTensor_free(state, grad_output);
  THCudaTensor_free(state, output);
  return 1;
}

int jagged_argmax_forward_cuda(THCudaTensor *values, THCudaLongTensor *prefix_sum, THCudaLongTensor *output)
{
  values = THCudaTensor_newContiguous(state, values);
  THCudaLongTensor_resizeAs(state, output, prefix_sum);
  
  float *input_data_base  = THCudaTensor_data(state, values);
  long* ps = THCudaLongTensor_data(state, prefix_sum);
  long *output_data_base  = THCudaLongTensor_data(state, output);
  
  int bsize = (int)prefix_sum->size[0];
  cudaStream_t stream = THCState_getCurrentStream(state);
  HostArgmaxForward(stream, input_data_base, output_data_base, ps, bsize); 

  THCudaTensor_free(state, values);
  return 1;
}

int jagged_max_forward_cuda(THCudaTensor *values, THCudaLongTensor *prefix_sum, THCudaTensor *vmax, THCudaLongTensor *idxes)
{
  int64_t inputsize = prefix_sum->size[0];
  values = THCudaTensor_newContiguous(state, values);
  THCudaLongTensor_resize1d(state, idxes, inputsize);
  THCudaTensor_resize1d(state, vmax, inputsize);

  float *input_data_base  = THCudaTensor_data(state, values);
  long* ps = THCudaLongTensor_data(state, prefix_sum);
  long *p_i  = THCudaLongTensor_data(state, idxes);
  float *p_maxv  = THCudaTensor_data(state, vmax);

  int bsize = (int)prefix_sum->size[0];
  cudaStream_t stream = THCState_getCurrentStream(state);
  HostMaxForward(stream, input_data_base, p_maxv, p_i, ps, bsize);

  THCudaTensor_free(state, values);
  return 1;  
}

int graph_laplacian_norm_cuda(THCudaLongTensor *indices, THCudaTensor *values, THCudaTensor *norm)
{
  uint64_t nnz = (uint64_t)values->size[0];
  long *row_indices = THCudaLongTensor_data(state, indices);
  long *col_indices = row_indices + THCudaLongTensor_stride(state, indices, 0);
  float *p_v = THCudaTensor_data(state, values);
  float *p_norm = THCudaTensor_data(state, norm);

  cudaStream_t stream = THCState_getCurrentStream(state);
  HostGLapNorm(stream, row_indices, col_indices, p_v, p_norm, nnz);
  return 1;
}

int graph_degree_norm_cuda(THCudaLongTensor *indices, THCudaTensor *values, THCudaTensor *norm)
{
  uint64_t nnz = (uint64_t)values->size[0];
  long *row_indices = THCudaLongTensor_data(state, indices);  
  float *p_v = THCudaTensor_data(state, values);
  float *p_norm = THCudaTensor_data(state, norm);

  cudaStream_t stream = THCState_getCurrentStream(state);
  HostGDegreeNorm(stream, row_indices, p_v, p_norm, nnz);
  return 1;  
}