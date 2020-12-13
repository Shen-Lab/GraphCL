#ifndef JAGGED_SOFTMAX_KERNEL_H
#define JAGGED_SOFTMAX_KERNEL_H

#ifdef __cplusplus
extern "C" {
#endif

void HostSoftMaxForward(cudaStream_t stream, float *input, float *output, long* ps, int bsize); 

void HostSoftMaxBackward(cudaStream_t stream, float *gradOutput, float *gradInput, float *output, long* ps, int bsize);

void HostArgmaxForward(cudaStream_t stream, float *input, long *output, long* ps, int bsize); 

void HostMaxForward(cudaStream_t stream, float *input, float* vmax, long *idxes, long* ps, int bsize); 

void HostGLapNorm(cudaStream_t stream, long* row_indices, long* col_indices, float* p_v, float* p_norm, int nnz);

void HostGDegreeNorm(cudaStream_t stream, long* row_indices, float* p_v, float* p_norm, int nnz);

#ifdef __cplusplus
}
#endif

#endif
