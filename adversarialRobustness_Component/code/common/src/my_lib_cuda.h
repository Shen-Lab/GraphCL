int jagged_log_softmax_forward_cuda(THCudaTensor *logits, THCudaLongTensor *prefix_sum, THCudaTensor *output);

int jagged_log_softmax_backward_cuda(THCudaTensor *output, THCudaTensor *grad_output, THCudaLongTensor *prefix_sum, THCudaTensor *grad_input);

int jagged_argmax_forward_cuda(THCudaTensor *values, THCudaLongTensor *prefix_sum, THCudaLongTensor *output);

int jagged_max_forward_cuda(THCudaTensor *values, THCudaLongTensor *prefix_sum, THCudaTensor *vmax, THCudaLongTensor *idxes);

int graph_laplacian_norm_cuda(THCudaLongTensor *indices, THCudaTensor *values, THCudaTensor *norm);

int graph_degree_norm_cuda(THCudaLongTensor *indices, THCudaTensor *values, THCudaTensor *norm);