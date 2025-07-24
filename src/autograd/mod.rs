use crate::tensor::Tensor;
use crate::error::{TorchError, Result};
use std::ffi::c_void;

extern "C" {
    fn tensor_requires_grad(tensor: *mut c_void, requires_grad: bool) -> *mut c_void;
    fn tensor_backward(tensor: *mut c_void) -> *mut c_void;
    fn tensor_grad(tensor: *mut c_void) -> *mut c_void;
}

impl Tensor {
    pub fn requires_grad(mut self, requires_grad: bool) -> Result<Self> {
        let ptr = unsafe { tensor_requires_grad(self.ptr, requires_grad) };
        
        if ptr.is_null() {
            return Err(TorchError::AutogradError("Setting requires_grad failed".to_string()));
        }
        
        self.ptr = ptr;
        Ok(self)
    }
    
    pub fn backward(&self) -> Result<()> {
        let result = unsafe { tensor_backward(self.ptr) };
        
        if !result.is_null() {
            return Err(TorchError::AutogradError("Backward pass failed".to_string()));
        }
        
        Ok(())
    }
    
    pub fn grad(&self) -> Result<Option<Tensor>> {
        let grad_ptr = unsafe { tensor_grad(self.ptr) };
        
        if grad_ptr.is_null() {
            Ok(None)
        } else {
            Ok(Some(Tensor {
                ptr: grad_ptr,
                shape: self.shape.clone(),
                dtype: self.dtype,
            }))
        }
    }
}

pub struct GradientContext {
    enabled: bool,
}

impl GradientContext {
    pub fn new() -> Self {
        GradientContext { enabled: true }
    }
    
    pub fn no_grad() -> Self {
        GradientContext { enabled: false }
    }
    
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }
}

impl Default for GradientContext {
    fn default() -> Self {
        Self::new()
    }
}

pub fn no_grad<F, R>(f: F) -> R
where
    F: FnOnce() -> R,
{
    let _context = GradientContext::no_grad();
    f()
}