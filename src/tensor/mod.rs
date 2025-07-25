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
    fn tensor_reshape(tensor: *mut c_void, sizes: *mut i64, ndim: i32) -> *mut c_void;
    fn tensor_clamp_min(tensor: *mut c_void, min_val: f32) -> *mut c_void;
    fn tensor_softmax(tensor: *mut c_void, dim: i64) -> *mut c_void;
    fn tensor_clone(tensor: *mut c_void) -> *mut c_void;
    fn tensor_requires_grad(tensor: *mut c_void, requires_grad: bool) -> *mut c_void;
    fn tensor_backward(tensor: *mut c_void);
    fn tensor_grad(tensor: *mut c_void) -> *mut c_void;
    fn tensor_sub(a: *mut c_void, b: *mut c_void) -> *mut c_void;
    fn tensor_pow(tensor: *mut c_void, exponent: f32) -> *mut c_void;
    fn tensor_mean(tensor: *mut c_void) -> *mut c_void;
}

#[derive(Debug, Clone, Copy)]
pub enum DType {
    Float32 = 6,
    Float64 = 7,
    Int32 = 3,
    Int64 = 4,
}

impl std::fmt::Display for DType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DType::Float32 => write!(f, "Float32"),
            DType::Float64 => write!(f, "Float64"),
            DType::Int32 => write!(f, "Int32"),
            DType::Int64 => write!(f, "Int64"),
        }
    }
}

pub struct Tensor {
    pub(crate) ptr: *mut c_void,
    pub(crate) shape: Vec<i64>,
    pub(crate) dtype: DType,
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
    
    pub fn reshape(&self, new_shape: Vec<i64>) -> Result<Tensor> {
        // Check if the new shape has the same number of elements
        let old_numel = self.numel();
        let new_numel: i64 = new_shape.iter().product();
        
        if old_numel != new_numel {
            return Err(TorchError::TensorError(format!(
                "Cannot reshape tensor with {} elements to shape with {} elements",
                old_numel, new_numel
            )));
        }
        
        let mut shape_array = new_shape.clone();
        let ptr = unsafe {
            tensor_reshape(self.ptr, shape_array.as_mut_ptr(), new_shape.len() as i32)
        };
        
        if ptr.is_null() {
            return Err(TorchError::TensorError("Reshape failed".to_string()));
        }
        
        Ok(Tensor {
            ptr,
            shape: new_shape,
            dtype: self.dtype,
        })
    }
    
    pub fn clamp_min(&self, min_val: f32) -> Result<Tensor> {
        let ptr = unsafe { tensor_clamp_min(self.ptr, min_val) };
        
        if ptr.is_null() {
            return Err(TorchError::TensorError("Clamp min failed".to_string()));
        }
        
        Ok(Tensor {
            ptr,
            shape: self.shape.clone(),
            dtype: self.dtype,
        })
    }
    
    pub fn softmax(&self, dim: i64) -> Result<Tensor> {
        let ptr = unsafe { tensor_softmax(self.ptr, dim) };
        
        if ptr.is_null() {
            return Err(TorchError::TensorError("Softmax failed".to_string()));
        }
        
        Ok(Tensor {
            ptr,
            shape: self.shape.clone(),
            dtype: self.dtype,
        })
    }
    
    pub fn requires_grad(self, requires_grad: bool) -> Result<Tensor> {
        let ptr = unsafe { tensor_requires_grad(self.ptr, requires_grad) };
        
        if ptr.is_null() {
            return Err(TorchError::AutogradError("Failed to set requires_grad".to_string()));
        }
        
        Ok(Tensor {
            ptr,
            shape: self.shape.clone(),
            dtype: self.dtype,
        })
    }
    
    pub fn backward(&self) -> Result<()> {
        unsafe { tensor_backward(self.ptr) };
        Ok(())
    }
    
    pub fn grad(&self) -> Result<Option<Tensor>> {
        let ptr = unsafe { tensor_grad(self.ptr) };
        
        if ptr.is_null() {
            Ok(None)
        } else {
            Ok(Some(Tensor {
                ptr,
                shape: self.shape.clone(),
                dtype: self.dtype,
            }))
        }
    }
    
    pub fn sub(&self, other: &Tensor) -> Result<Tensor> {
        if self.shape != other.shape {
            return Err(TorchError::ShapeMismatch {
                expected: self.shape.clone(),
                actual: other.shape.clone(),
            });
        }
        
        let ptr = unsafe { tensor_sub(self.ptr, other.ptr) };
        
        if ptr.is_null() {
            return Err(TorchError::TensorError("Subtraction failed".to_string()));
        }
        
        Ok(Tensor {
            ptr,
            shape: self.shape.clone(),
            dtype: self.dtype,
        })
    }
    
    pub fn pow(&self, exponent: f32) -> Result<Tensor> {
        let ptr = unsafe { tensor_pow(self.ptr, exponent) };
        
        if ptr.is_null() {
            return Err(TorchError::TensorError("Power operation failed".to_string()));
        }
        
        Ok(Tensor {
            ptr,
            shape: self.shape.clone(),
            dtype: self.dtype,
        })
    }
    
    pub fn mean(&self) -> Result<Tensor> {
        let ptr = unsafe { tensor_mean(self.ptr) };
        
        if ptr.is_null() {
            return Err(TorchError::TensorError("Mean operation failed".to_string()));
        }
        
        Ok(Tensor {
            ptr,
            shape: vec![1], // Mean returns a scalar
            dtype: self.dtype,
        })
    }
}

impl Clone for Tensor {
    fn clone(&self) -> Self {
        let ptr = unsafe { tensor_clone(self.ptr) };
        
        if ptr.is_null() {
            panic!("Failed to clone tensor");
        }
        
        Tensor {
            ptr,
            shape: self.shape.clone(),
            dtype: self.dtype,
        }
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