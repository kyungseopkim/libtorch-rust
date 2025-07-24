use crate::tensor::Tensor;
use crate::error::{TorchError, Result};
use std::ffi::c_void;

extern "C" {
    fn tensor_cuda(tensor: *mut c_void) -> *mut c_void;
    fn tensor_cpu(tensor: *mut c_void) -> *mut c_void;
    fn tensor_is_cuda(tensor: *mut c_void) -> bool;
}

impl Tensor {
    pub fn cuda(mut self) -> Result<Self> {
        #[cfg(not(feature = "cuda"))]
        {
            return Err(TorchError::CudaError("CUDA support not enabled".to_string()));
        }
        
        #[cfg(feature = "cuda")]
        {
            let ptr = unsafe { tensor_cuda(self.ptr) };
            
            if ptr.is_null() {
                return Err(TorchError::CudaError("Failed to move tensor to CUDA".to_string()));
            }
            
            self.ptr = ptr;
            Ok(self)
        }
    }
    
    pub fn cpu(mut self) -> Result<Self> {
        let ptr = unsafe { tensor_cpu(self.ptr) };
        
        if ptr.is_null() {
            return Err(TorchError::CudaError("Failed to move tensor to CPU".to_string()));
        }
        
        self.ptr = ptr;
        Ok(self)
    }
    
    pub fn is_cuda(&self) -> bool {
        unsafe { tensor_is_cuda(self.ptr) }
    }
    
    pub fn device(&self) -> Device {
        if self.is_cuda() {
            Device::Cuda(0)
        } else {
            Device::Cpu
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Device {
    Cpu,
    Cuda(i32),
}

impl Device {
    pub fn cuda(index: i32) -> Self {
        Device::Cuda(index)
    }
    
    pub fn cpu() -> Self {
        Device::Cpu
    }
    
    pub fn is_cuda(&self) -> bool {
        matches!(self, Device::Cuda(_))
    }
    
    pub fn index(&self) -> Option<i32> {
        match self {
            Device::Cuda(index) => Some(*index),
            Device::Cpu => None,
        }
    }
}

impl std::fmt::Display for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Device::Cpu => write!(f, "cpu"),
            Device::Cuda(index) => write!(f, "cuda:{}", index),
        }
    }
}

pub fn is_available() -> bool {
    #[cfg(feature = "cuda")]
    {
        true
    }
    #[cfg(not(feature = "cuda"))]
    {
        false
    }
}

pub fn device_count() -> i32 {
    #[cfg(feature = "cuda")]
    {
        1
    }
    #[cfg(not(feature = "cuda"))]
    {
        0
    }
}