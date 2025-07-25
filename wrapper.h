#ifndef WRAPPER_H
#define WRAPPER_H

#include <torch/torch.h>
#include <torch/script.h>

#ifdef __cplusplus
extern "C" {
#endif

// Tensor operations
void* tensor_new_f32(float* data, long* sizes, int ndim);
void* tensor_new_empty(long* sizes, int ndim, int dtype);
void tensor_delete(void* tensor);
float* tensor_data_ptr_f32(void* tensor);
void tensor_print(void* tensor);
void* tensor_add(void* a, void* b);
void* tensor_mul(void* a, void* b);
void* tensor_matmul(void* a, void* b);
void* tensor_reshape(void* tensor, long* sizes, int ndim);
void* tensor_clamp_min(void* tensor, float min_val);
void* tensor_softmax(void* tensor, long dim);
void* tensor_clone(void* tensor);
void* tensor_sub(void* a, void* b);
void* tensor_pow(void* tensor, float exponent);
void* tensor_mean(void* tensor);

// Autograd operations  
void* tensor_requires_grad(void* tensor, bool requires_grad);
void tensor_backward(void* tensor);
void* tensor_grad(void* tensor);

// Neural network operations
void* linear_new(int in_features, int out_features);
void linear_delete(void* linear);
void* linear_forward(void* linear, void* input);


// CUDA operations
void* tensor_cuda(void* tensor);
void* tensor_cpu(void* tensor);
bool tensor_is_cuda(void* tensor);

#ifdef __cplusplus
}
#endif

#endif