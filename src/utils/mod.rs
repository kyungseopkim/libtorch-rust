use crate::tensor::{Tensor, DType};
use crate::error::Result;

pub fn randn(shape: Vec<i64>) -> Result<Tensor> {
    use std::f32::consts::PI;
    
    let size: usize = shape.iter().product::<i64>() as usize;
    let mut data = Vec::with_capacity(size);
    
    for i in 0..size {
        let u1 = (i as f32 + 1.0) / (size as f32 + 1.0);
        let u2 = (i as f32 * 2.0 + 1.0) / (size as f32 * 2.0 + 1.0);
        
        let z0 = (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos();
        data.push(z0);
    }
    
    Tensor::new(data, shape)
}

pub fn rand(shape: Vec<i64>) -> Result<Tensor> {
    let size: usize = shape.iter().product::<i64>() as usize;
    let mut data = Vec::with_capacity(size);
    
    for i in 0..size {
        let val = (i as f32 % 1000.0) / 1000.0;
        data.push(val);
    }
    
    Tensor::new(data, shape)
}

pub fn arange(start: f32, end: f32, step: f32) -> Result<Tensor> {
    let mut data = Vec::new();
    let mut current = start;
    
    while current < end {
        data.push(current);
        current += step;
    }
    
    let shape = vec![data.len() as i64];
    Tensor::new(data, shape)
}

pub fn linspace(start: f32, end: f32, steps: i64) -> Result<Tensor> {
    if steps <= 1 {
        return Tensor::new(vec![start], vec![1]);
    }
    
    let mut data = Vec::with_capacity(steps as usize);
    let step_size = (end - start) / (steps - 1) as f32;
    
    for i in 0..steps {
        data.push(start + i as f32 * step_size);
    }
    
    let shape = vec![steps];
    Tensor::new(data, shape)
}

pub fn eye(n: i64) -> Result<Tensor> {
    let size = (n * n) as usize;
    let mut data = vec![0.0f32; size];
    
    for i in 0..n {
        data[(i * n + i) as usize] = 1.0;
    }
    
    Tensor::new(data, vec![n, n])
}

pub trait TensorUtils {
    fn numel(&self) -> i64;
    fn size(&self) -> &[i64];
    fn dim(&self) -> usize;
    fn is_contiguous(&self) -> bool;
}

impl TensorUtils for Tensor {
    fn numel(&self) -> i64 {
        self.numel()
    }
    
    fn size(&self) -> &[i64] {
        self.shape()
    }
    
    fn dim(&self) -> usize {
        self.shape().len()
    }
    
    fn is_contiguous(&self) -> bool {
        true
    }
}

pub fn save_tensor(_tensor: &Tensor, _path: &str) -> Result<()> {
    Ok(())
}

pub fn load_tensor(_path: &str) -> Result<Tensor> {
    Tensor::zeros(vec![1])
}