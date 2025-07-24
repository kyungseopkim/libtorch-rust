use libtorch_rust::{Tensor, DType, Result};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_creation() -> Result<()> {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let tensor = Tensor::new(data, vec![2, 2])?;
        
        assert_eq!(tensor.shape(), &[2, 2]);
        assert_eq!(tensor.numel(), 4);
        
        Ok(())
    }
    
    #[test]
    fn test_zeros_ones() -> Result<()> {
        let zeros = Tensor::zeros(vec![3, 3])?;
        assert_eq!(zeros.shape(), &[3, 3]);
        assert_eq!(zeros.numel(), 9);
        
        let ones = Tensor::ones(vec![2, 4])?;
        assert_eq!(ones.shape(), &[2, 4]);
        assert_eq!(ones.numel(), 8);
        
        Ok(())
    }
    
    #[test]
    fn test_tensor_operations() -> Result<()> {
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2])?;
        let b = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2])?;
        
        let sum = a.add(&b)?;
        assert_eq!(sum.shape(), &[2, 2]);
        
        let product = a.mul(&b)?;
        assert_eq!(product.shape(), &[2, 2]);
        
        let matmul_result = a.matmul(&b)?;
        assert_eq!(matmul_result.shape(), &[2, 2]);
        
        Ok(())
    }
    
    #[test]
    fn test_tensor_reshape() -> Result<()> {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = Tensor::new(data, vec![2, 3])?;
        
        let reshaped = tensor.reshape(vec![3, 2])?;
        assert_eq!(reshaped.shape(), &[3, 2]);
        assert_eq!(reshaped.numel(), 6);
        
        Ok(())
    }
    
    #[test]
    fn test_activation_functions() -> Result<()> {
        let data = vec![-1.0, 0.0, 1.0, 2.0];
        let tensor = Tensor::new(data, vec![4])?;
        
        let relu_result = tensor.clamp_min(0.0)?;
        assert_eq!(relu_result.shape(), &[4]);
        
        let sigmoid_result = tensor.sigmoid()?;
        assert_eq!(sigmoid_result.shape(), &[4]);
        
        let softmax_result = tensor.softmax(0)?;
        assert_eq!(softmax_result.shape(), &[4]);
        
        Ok(())
    }
}