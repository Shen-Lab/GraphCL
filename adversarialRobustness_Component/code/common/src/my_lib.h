int jagged_log_softmax_forward(THFloatTensor *logits, THLongTensor *prefix_sum, THFloatTensor *output);

int jagged_log_softmax_backward(THFloatTensor *output, THFloatTensor *grad_output, THLongTensor *prefix_sum, THFloatTensor *grad_input);

int jagged_argmax_forward(THFloatTensor *values, THLongTensor *prefix_sum, THLongTensor *output);

int jagged_max_forward(THFloatTensor *values, THLongTensor *prefix_sum, THFloatTensor *vmax, THLongTensor *idxes);

int graph_laplacian_norm(THLongTensor *indices, THFloatTensor *values, THFloatTensor *norm);

int graph_degree_norm(THLongTensor *indices, THFloatTensor *values, THFloatTensor *norm);