// Simple test runner to verify LibTorch Rust bindings functionality
use crate::{Tensor, Result};

pub fn run_basic_tests() -> Result<()> {
    println!("🧪 Running LibTorch Rust Basic Tests");
    println!("====================================");
    
    // Test 1: Tensor creation
    println!("\n1. Testing tensor creation...");
    let data = vec![1.0, 2.0, 3.0, 4.0];
    let tensor = Tensor::new(data, vec![2, 2])?;
    assert_eq!(tensor.shape(), &[2, 2]);
    assert_eq!(tensor.numel(), 4);
    println!("   ✅ Tensor creation test passed");
    
    // Test 2: Zeros and ones
    println!("\n2. Testing zeros and ones tensors...");
    let zeros = Tensor::zeros(vec![3, 3])?;
    assert_eq!(zeros.shape(), &[3, 3]);
    assert_eq!(zeros.numel(), 9);
    
    let ones = Tensor::ones(vec![2, 4])?;
    assert_eq!(ones.shape(), &[2, 4]);
    assert_eq!(ones.numel(), 8);
    println!("   ✅ Zeros and ones test passed");
    
    // Test 3: Basic operations
    println!("\n3. Testing tensor operations...");
    let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2])?;
    let b = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2])?;
    
    let sum = a.add(&b)?;
    assert_eq!(sum.shape(), &[2, 2]);
    
    let product = a.mul(&b)?;
    assert_eq!(product.shape(), &[2, 2]);
    
    let matmul_result = a.matmul(&b)?;
    assert_eq!(matmul_result.shape(), &[2, 2]);
    println!("   ✅ Tensor operations test passed");
    
    // Test 4: Reshape
    println!("\n4. Testing tensor reshape...");
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let tensor = Tensor::new(data, vec![2, 3])?;
    
    let reshaped = tensor.reshape(vec![3, 2])?;
    assert_eq!(reshaped.shape(), &[3, 2]);
    assert_eq!(reshaped.numel(), 6);
    println!("   ✅ Tensor reshape test passed");
    
    // Test 5: Data type and device info
    println!("\n5. Testing tensor properties...");
    let tensor = Tensor::new(vec![1.0, 2.0], vec![2])?;
    assert_eq!(tensor.dtype().to_string(), "Float32");
    println!("   ✅ Tensor properties test passed");
    
    println!("\n🎉 All basic tests passed successfully!");
    println!("📊 Test summary:");
    println!("   • Tensor creation: ✅");
    println!("   • Special tensors (zeros/ones): ✅");
    println!("   • Arithmetic operations: ✅");
    println!("   • Matrix multiplication: ✅");
    println!("   • Tensor reshaping: ✅");
    println!("   • Tensor properties: ✅");
    
    Ok(())
}

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
}