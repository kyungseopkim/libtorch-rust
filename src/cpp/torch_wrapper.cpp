#include "../../wrapper.h"
#include <torch/torch.h>
#include <iostream>

extern "C" {

void* tensor_new_f32(float* data, long* sizes, int ndim) {
    std::vector<int64_t> shape(sizes, sizes + ndim);
    torch::Tensor* tensor = new torch::Tensor(torch::from_blob(data, shape, torch::kFloat32).clone());
    return static_cast<void*>(tensor);
}

void* tensor_new_empty(long* sizes, int ndim, int dtype) {
    std::vector<int64_t> shape(sizes, sizes + ndim);
    torch::ScalarType scalar_type = static_cast<torch::ScalarType>(dtype);
    torch::Tensor* tensor = new torch::Tensor(torch::empty(shape, scalar_type));
    return static_cast<void*>(tensor);
}

void tensor_delete(void* tensor) {
    delete static_cast<torch::Tensor*>(tensor);
}

float* tensor_data_ptr_f32(void* tensor) {
    torch::Tensor* t = static_cast<torch::Tensor*>(tensor);
    return t->data_ptr<float>();
}

void tensor_print(void* tensor) {
    torch::Tensor* t = static_cast<torch::Tensor*>(tensor);
    std::cout << *t << std::endl;
}

void* tensor_add(void* a, void* b) {
    torch::Tensor* ta = static_cast<torch::Tensor*>(a);
    torch::Tensor* tb = static_cast<torch::Tensor*>(b);
    torch::Tensor* result = new torch::Tensor(*ta + *tb);
    return static_cast<void*>(result);
}

void* tensor_mul(void* a, void* b) {
    torch::Tensor* ta = static_cast<torch::Tensor*>(a);
    torch::Tensor* tb = static_cast<torch::Tensor*>(b);
    torch::Tensor* result = new torch::Tensor(*ta * *tb);
    return static_cast<void*>(result);
}

void* tensor_matmul(void* a, void* b) {
    torch::Tensor* ta = static_cast<torch::Tensor*>(a);
    torch::Tensor* tb = static_cast<torch::Tensor*>(b);
    torch::Tensor* result = new torch::Tensor(torch::matmul(*ta, *tb));
    return static_cast<void*>(result);
}

void* linear_new(int in_features, int out_features) {
    torch::nn::Linear linear = torch::nn::Linear(torch::nn::LinearOptions(in_features, out_features));
    torch::nn::Linear* linear_ptr = new torch::nn::Linear(linear);
    return static_cast<void*>(linear_ptr);
}

void linear_delete(void* linear) {
    delete static_cast<torch::nn::Linear*>(linear);
}

void* linear_forward(void* linear, void* input) {
    torch::nn::Linear* l = static_cast<torch::nn::Linear*>(linear);
    torch::Tensor* in = static_cast<torch::Tensor*>(input);
    torch::Tensor result = (*l)(*in);
    torch::Tensor* result_ptr = new torch::Tensor(result);
    return static_cast<void*>(result_ptr);
}

void* tensor_requires_grad(void* tensor, bool requires_grad) {
    torch::Tensor* t = static_cast<torch::Tensor*>(tensor);
    torch::Tensor* result = new torch::Tensor(t->requires_grad_(requires_grad));
    return static_cast<void*>(result);
}

void* tensor_backward(void* tensor) {
    torch::Tensor* t = static_cast<torch::Tensor*>(tensor);
    t->backward();
    return nullptr;
}

void* tensor_grad(void* tensor) {
    torch::Tensor* t = static_cast<torch::Tensor*>(tensor);
    if (t->grad().defined()) {
        torch::Tensor* grad = new torch::Tensor(t->grad());
        return static_cast<void*>(grad);
    }
    return nullptr;
}

void* tensor_cuda(void* tensor) {
    torch::Tensor* t = static_cast<torch::Tensor*>(tensor);
    torch::Tensor* result = new torch::Tensor(t->cuda());
    return static_cast<void*>(result);
}

void* tensor_cpu(void* tensor) {
    torch::Tensor* t = static_cast<torch::Tensor*>(tensor);
    torch::Tensor* result = new torch::Tensor(t->cpu());
    return static_cast<void*>(result);
}

bool tensor_is_cuda(void* tensor) {
    torch::Tensor* t = static_cast<torch::Tensor*>(tensor);
    return t->is_cuda();
}

}