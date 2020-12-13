#include <TH/TH.h>
#include <assert.h>

int jagged_argmax_forward(THFloatTensor *values, THLongTensor *prefix_sum, THLongTensor *output)
{
  values = THFloatTensor_newContiguous(values);
  THLongTensor_resizeAs(output, prefix_sum);

  float *input_data_base = values->storage->data + values->storageOffset;;  
  long *ps = prefix_sum->storage->data + prefix_sum->storageOffset;
  long *p_out = output->storage->data + output->storageOffset;
  long bsize = (long)prefix_sum->size[0];
  long i, d;

  #pragma omp parallel for private(i, d)
  for (i = 0; i < bsize; i++)
  {
    long offset = (i == 0) ? 0 : ps[i - 1];
    long n_ele = ps[i] - offset;

    float* input_data  = input_data_base  + offset;

    float max_input = -FLT_MAX;
    long max_id = -1;
    for (d = 0; d < n_ele; d++)
      if (input_data[d] > max_input)
      {
        max_input = input_data[d];
        max_id = d;
      }
    assert(max_id >= 0);
    p_out[i] = max_id;
  }

  THFloatTensor_free(values);
  return 1;
}

int jagged_max_forward(THFloatTensor *values, THLongTensor *prefix_sum, THFloatTensor *vmax, THLongTensor *idxes)
{
  int64_t inputsize = prefix_sum->size[0];

  values = THFloatTensor_newContiguous(values);
  THLongTensor_resize1d(idxes, inputsize);
  THFloatTensor_resize1d(vmax, inputsize);

  float *input_data_base = values->storage->data + values->storageOffset;
  long *ps = prefix_sum->storage->data + prefix_sum->storageOffset;
  float *p_maxv = vmax->storage->data + vmax->storageOffset;
  long *p_i = idxes->storage->data + idxes->storageOffset;

  long bsize = (long)prefix_sum->size[0];
  long i, d;

  #pragma omp parallel for private(i, d)
  for (i = 0; i < bsize; i++)
  {
    long offset = (i == 0) ? 0 : ps[i - 1];
    long n_ele = ps[i] - offset;

    float* input_data  = input_data_base  + offset;

    float max_input = -FLT_MAX;
    long max_id = -1;
    for (d = 0; d < n_ele; d++)
      if (input_data[d] > max_input)
      {
        max_input = input_data[d];
        max_id = d;
      }
    assert(max_id >= 0);
    p_i[i] = max_id;
    p_maxv[i] = max_input;
  }

  THFloatTensor_free(values);
  return 1;  
}

int jagged_log_softmax_forward(THFloatTensor *logits, THLongTensor *prefix_sum, THFloatTensor *output)
{
  logits = THFloatTensor_newContiguous(logits);
  THFloatTensor_resizeAs(output, logits);   
  float *input_data_base  = logits->storage->data + logits->storageOffset;//  THTensor_(data)(logits);  
  long *ps = prefix_sum->storage->data + prefix_sum->storageOffset;
  float *output_data_base = output->storage->data + output->storageOffset;
  uint64_t bsize = (uint64_t)prefix_sum->size[0];
  uint64_t i, d;

  #pragma omp parallel for private(i, d)
  for (i = 0; i < bsize; i++)
  {
    long offset = (i == 0) ? 0 : ps[i - 1];

    float* input_data  = input_data_base  + offset;
    float* output_data = output_data_base + offset;

    long n_ele = ps[i] - offset;
    float max_input = -FLT_MAX;
    for (d = 0; d < n_ele; d++)
      max_input = THMax(max_input, input_data[d]);

    double logsum = 0;
    for (d = 0; d < n_ele; d++)
      logsum += exp(input_data[d] - max_input);
    logsum = max_input + log(logsum);

    for (d = 0; d < n_ele; d++)
      output_data[d] = input_data[d] - logsum;
  }

  THFloatTensor_free(logits);
  return 1;
}

int jagged_log_softmax_backward(THFloatTensor *output, THFloatTensor *grad_output, THLongTensor *prefix_sum, THFloatTensor *grad_input)
{
  grad_output = THFloatTensor_newContiguous(grad_output);
  output = THFloatTensor_newContiguous(output); 
  THFloatTensor_resizeAs(grad_input, grad_output); 
  
  float *output_data_base = output->storage->data + output->storageOffset;
  float *gradOutput_data_base  = grad_output->storage->data + grad_output->storageOffset; 
  long *ps = prefix_sum->storage->data + prefix_sum->storageOffset;
  float *gradInput_data_base  = grad_input->storage->data + grad_input->storageOffset; 
  
  uint64_t bsize = (uint64_t)prefix_sum->size[0];
  uint64_t i, d;
  #pragma omp parallel for private(i, d)
  for (i = 0; i < bsize; i++)
  {
    long offset = (i == 0) ? 0 : ps[i - 1];
    float *gradInput_data  = gradInput_data_base  + offset;
    float *output_data     = output_data_base     + offset;
    float *gradOutput_data = gradOutput_data_base + offset;

    double sum = 0;
    long n_ele = ps[i] - offset;
    for (d = 0; d < n_ele; d++)
      sum += gradOutput_data[d];

    for (d = 0; d < n_ele; d++)
      gradInput_data[d] = gradOutput_data[d] - exp(output_data[d]) * sum;
  }

  THFloatTensor_free(grad_output);
  THFloatTensor_free(output);
  return 1;
}

int graph_laplacian_norm(THLongTensor *indices, THFloatTensor *values, THFloatTensor *norm)
{
  uint64_t nnz = (uint64_t)values->size[0];
  long *row_indices = indices->storage->data + indices->storageOffset;
  long *col_indices = row_indices + indices->stride[0];
  float *p_v = values->storage->data + values->storageOffset;
  float *p_norm = norm->storage->data + norm->storageOffset;

  uint64_t i;
  #pragma omp parallel for private(i)  
  for (i = 0; i < nnz; i++)
  {    
    float norm = p_norm[ row_indices[i] ] * p_norm[ col_indices[i] ];
    p_v[i] /= norm;
  }

  return 1;
}

int graph_degree_norm(THLongTensor *indices, THFloatTensor *values, THFloatTensor *norm)
{
  uint64_t nnz = (uint64_t)values->size[0];
  long *row_indices = indices->storage->data + indices->storageOffset;
  float *p_v = values->storage->data + values->storageOffset;
  float *p_norm = norm->storage->data + norm->storageOffset;

  uint64_t i;
  #pragma omp parallel for private(i)
  for (i = 0; i < nnz; i++)
  {
    float norm = p_norm[ row_indices[i] ];
    p_v[i] /= norm;
  }

  return 1;  
}