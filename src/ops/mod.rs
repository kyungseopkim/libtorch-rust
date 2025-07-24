use crate::tensor::Tensor;
use crate::error::Result;

impl Tensor {
    pub fn clamp_min(&self, min: f32) -> Result<Tensor> {
        let data_ptr = self.data_ptr();
        let size = self.numel() as usize;
        
        let mut new_data = Vec::with_capacity(size);
        unsafe {
            for i in 0..size {
                let val = *data_ptr.add(i);
                new_data.push(val.max(min));
            }
        }
        
        Self::new(new_data, self.shape.clone())
    }
    
    pub fn sigmoid(&self) -> Result<Tensor> {
        let data_ptr = self.data_ptr();
        let size = self.numel() as usize;
        
        let mut new_data = Vec::with_capacity(size);
        unsafe {
            for i in 0..size {
                let val = *data_ptr.add(i);
                new_data.push(1.0 / (1.0 + (-val).exp()));
            }
        }
        
        Self::new(new_data, self.shape.clone())
    }
    
    pub fn softmax(&self, _dim: i64) -> Result<Tensor> {
        let data_ptr = self.data_ptr();
        let size = self.numel() as usize;
        
        let mut new_data = Vec::with_capacity(size);
        let mut max_val = f32::NEG_INFINITY;
        
        unsafe {
            for i in 0..size {
                let val = *data_ptr.add(i);
                max_val = max_val.max(val);
            }
            
            let mut sum = 0.0;
            for i in 0..size {
                let val = *data_ptr.add(i);
                let exp_val = (val - max_val).exp();
                new_data.push(exp_val);
                sum += exp_val;
            }
            
            for val in &mut new_data {
                *val /= sum;
            }
        }
        
        Self::new(new_data, self.shape.clone())
    }
    
    pub fn log_softmax(&self, dim: i64) -> Result<Tensor> {
        let softmax_result = self.softmax(dim)?;
        softmax_result.log()
    }
    
    pub fn log(&self) -> Result<Tensor> {
        let data_ptr = self.data_ptr();
        let size = self.numel() as usize;
        
        let mut new_data = Vec::with_capacity(size);
        unsafe {
            for i in 0..size {
                let val = *data_ptr.add(i);
                new_data.push(val.ln());
            }
        }
        
        Self::new(new_data, self.shape.clone())
    }
    
    pub fn pow(&self, exponent: f32) -> Result<Tensor> {
        let data_ptr = self.data_ptr();
        let size = self.numel() as usize;
        
        let mut new_data = Vec::with_capacity(size);
        unsafe {
            for i in 0..size {
                let val = *data_ptr.add(i);
                new_data.push(val.powf(exponent));
            }
        }
        
        Self::new(new_data, self.shape.clone())
    }
    
    pub fn sub(&self, other: &Tensor) -> Result<Tensor> {
        let data_ptr_a = self.data_ptr();
        let data_ptr_b = other.data_ptr();
        let size = self.numel() as usize;
        
        let mut new_data = Vec::with_capacity(size);
        unsafe {
            for i in 0..size {
                let val_a = *data_ptr_a.add(i);
                let val_b = *data_ptr_b.add(i);
                new_data.push(val_a - val_b);
            }
        }
        
        Self::new(new_data, self.shape.clone())
    }
    
    pub fn mean(&self) -> Result<Tensor> {
        let data_ptr = self.data_ptr();
        let size = self.numel() as usize;
        
        let mut sum = 0.0;
        unsafe {
            for i in 0..size {
                sum += *data_ptr.add(i);
            }
        }
        
        let mean_val = sum / size as f32;
        Self::new(vec![mean_val], vec![1])
    }
    
    pub fn nll_loss(&self, _target: &Tensor) -> Result<Tensor> {
        Self::new(vec![0.0], vec![1])
    }
    
    pub fn reshape(&self, shape: Vec<i64>) -> Result<Tensor> {
        let total_elements: i64 = shape.iter().product();
        if total_elements != self.numel() {
            return Err(crate::error::TorchError::TensorError(
                "Cannot reshape tensor: total elements don't match".to_string()
            ));
        }
        
        let data_ptr = self.data_ptr();
        let size = self.numel() as usize;
        
        let mut new_data = Vec::with_capacity(size);
        unsafe {
            for i in 0..size {
                new_data.push(*data_ptr.add(i));
            }
        }
        
        Self::new(new_data, shape)
    }
}