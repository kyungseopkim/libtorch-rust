use crate::tensor::Tensor;
use crate::error::{TorchError, Result};
use std::ffi::c_void;

extern "C" {
    fn linear_new(in_features: i32, out_features: i32) -> *mut c_void;
    fn linear_delete(linear: *mut c_void);
    fn linear_forward(linear: *mut c_void, input: *mut c_void) -> *mut c_void;
}

pub trait Module {
    fn forward(&self, input: &Tensor) -> Result<Tensor>;
}

pub struct Linear {
    ptr: *mut c_void,
    in_features: i32,
    out_features: i32,
}

impl Linear {
    pub fn new(in_features: i32, out_features: i32) -> Result<Self> {
        let ptr = unsafe { linear_new(in_features, out_features) };
        
        if ptr.is_null() {
            return Err(TorchError::NeuralNetworkError("Failed to create Linear layer".to_string()));
        }
        
        Ok(Linear {
            ptr,
            in_features,
            out_features,
        })
    }
    
    pub fn in_features(&self) -> i32 {
        self.in_features
    }
    
    pub fn out_features(&self) -> i32 {
        self.out_features
    }
}

impl Module for Linear {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let output_ptr = unsafe { linear_forward(self.ptr, input.ptr as *mut c_void) };
        
        if output_ptr.is_null() {
            return Err(TorchError::NeuralNetworkError("Linear forward pass failed".to_string()));
        }
        
        let output_shape = if input.shape().len() == 2 {
            vec![input.shape()[0], self.out_features as i64]
        } else {
            vec![self.out_features as i64]
        };
        
        Ok(Tensor {
            ptr: output_ptr,
            shape: output_shape,
            dtype: input.dtype(),
        })
    }
}

impl Drop for Linear {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { linear_delete(self.ptr) };
        }
    }
}

unsafe impl Send for Linear {}
unsafe impl Sync for Linear {}

pub mod activation {
    use crate::tensor::Tensor;
    use crate::error::Result;
    
    pub fn relu(input: &Tensor) -> Result<Tensor> {
        input.clamp_min(0.0)
    }
    
    pub fn softmax(input: &Tensor, dim: i64) -> Result<Tensor> {
        input.softmax(dim)
    }
}