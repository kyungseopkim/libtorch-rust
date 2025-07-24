use thiserror::Error;

#[derive(Error, Debug)]
pub enum TorchError {
    #[error("Tensor operation failed: {0}")]
    TensorError(String),
    
    #[error("Neural network error: {0}")]
    NeuralNetworkError(String),
    
    #[error("CUDA error: {0}")]
    CudaError(String),
    
    #[error("Autograd error: {0}")]
    AutogradError(String),
    
    #[error("Shape mismatch: expected {expected:?}, got {actual:?}")]
    ShapeMismatch { expected: Vec<i64>, actual: Vec<i64> },
    
    #[error("Invalid dtype: {0}")]
    InvalidDtype(String),
    
    #[error("Memory allocation failed")]
    MemoryError,
    
    #[error("LibTorch C++ error: {0}")]
    CppError(String),
}

pub type Result<T> = std::result::Result<T, TorchError>;