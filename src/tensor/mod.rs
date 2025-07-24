use crate::error::{TorchError, Result};
use std::ffi::c_void;

extern "C" {
    fn tensor_new_f32(data: *mut f32, sizes: *mut i64, ndim: i32) -> *mut c_void;
    fn tensor_new_empty(sizes: *mut i64, ndim: i32, dtype: i32) -> *mut c_void;
    fn tensor_delete(tensor: *mut c_void);
    fn tensor_data_ptr_f32(tensor: *mut c_void) -> *mut f32;
    fn tensor_print(tensor: *mut c_void);
    fn tensor_add(a: *mut c_void, b: *mut c_void) -> *mut c_void;
    fn tensor_mul(a: *mut c_void, b: *mut c_void) -> *mut c_void;
    fn tensor_matmul(a: *mut c_void, b: *mut c_void) -> *mut c_void;
}

#[derive(Debug, Clone, Copy)]
pub enum DType {
    Float32 = 6,
    Float64 = 7,
    Int32 = 3,
    Int64 = 4,
}

pub struct Tensor {
    pub(crate) ptr: *mut c_void,
    shape: Vec<i64>,
    dtype: DType,
}

impl Tensor {
    pub fn new(data: Vec<f32>, shape: Vec<i64>) -> Result<Self> {
        let mut data = data;
        let mut shape_array: Vec<i64> = shape.clone();
        
        let ptr = unsafe {
            tensor_new_f32(
                data.as_mut_ptr(),
                shape_array.as_mut_ptr(),
                shape.len() as i32,
            )
        };
        
        if ptr.is_null() {
            return Err(TorchError::TensorError("Failed to create tensor".to_string()));
        }
        
        Ok(Tensor {
            ptr,
            shape,
            dtype: DType::Float32,
        })
    }
    
    pub fn empty(shape: Vec<i64>, dtype: DType) -> Result<Self> {
        let mut shape_array = shape.clone();
        
        let ptr = unsafe {
            tensor_new_empty(
                shape_array.as_mut_ptr(),
                shape.len() as i32,
                dtype as i32,
            )
        };
        
        if ptr.is_null() {
            return Err(TorchError::TensorError("Failed to create empty tensor".to_string()));
        }
        
        Ok(Tensor { ptr, shape, dtype })
    }
    
    pub fn zeros(shape: Vec<i64>) -> Result<Self> {
        let size: usize = shape.iter().product::<i64>() as usize;
        let data = vec![0.0f32; size];
        Self::new(data, shape)
    }
    
    pub fn ones(shape: Vec<i64>) -> Result<Self> {
        let size: usize = shape.iter().product::<i64>() as usize;
        let data = vec![1.0f32; size];
        Self::new(data, shape)
    }
    
    pub fn shape(&self) -> &[i64] {
        &self.shape
    }
    
    pub fn dtype(&self) -> DType {
        self.dtype
    }
    
    pub fn numel(&self) -> i64 {
        self.shape.iter().product()
    }
    
    pub fn data_ptr(&self) -> *mut f32 {
        unsafe { tensor_data_ptr_f32(self.ptr) }
    }
    
    pub fn print(&self) {
        unsafe { tensor_print(self.ptr) };
    }
    
    pub fn add(&self, other: &Tensor) -> Result<Tensor> {
        if self.shape != other.shape {
            return Err(TorchError::ShapeMismatch {
                expected: self.shape.clone(),
                actual: other.shape.clone(),
            });
        }
        
        let ptr = unsafe { tensor_add(self.ptr, other.ptr) };
        
        if ptr.is_null() {
            return Err(TorchError::TensorError("Addition failed".to_string()));
        }
        
        Ok(Tensor {
            ptr,
            shape: self.shape.clone(),
            dtype: self.dtype,
        })
    }
    
    pub fn mul(&self, other: &Tensor) -> Result<Tensor> {
        if self.shape != other.shape {
            return Err(TorchError::ShapeMismatch {
                expected: self.shape.clone(),
                actual: other.shape.clone(),
            });
        }
        
        let ptr = unsafe { tensor_mul(self.ptr, other.ptr) };
        
        if ptr.is_null() {
            return Err(TorchError::TensorError("Multiplication failed".to_string()));
        }
        
        Ok(Tensor {
            ptr,
            shape: self.shape.clone(),
            dtype: self.dtype,
        })
    }
    
    pub fn matmul(&self, other: &Tensor) -> Result<Tensor> {
        let ptr = unsafe { tensor_matmul(self.ptr, other.ptr) };
        
        if ptr.is_null() {
            return Err(TorchError::TensorError("Matrix multiplication failed".to_string()));
        }
        
        let result_shape = if self.shape.len() == 2 && other.shape.len() == 2 {
            vec![self.shape[0], other.shape[1]]
        } else {
            self.shape.clone()
        };
        
        Ok(Tensor {
            ptr,
            shape: result_shape,
            dtype: self.dtype,
        })
    }
}

impl Drop for Tensor {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { tensor_delete(self.ptr) };
        }
    }
}

unsafe impl Send for Tensor {}
unsafe impl Sync for Tensor {}