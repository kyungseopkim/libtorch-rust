use libtorch_rust::{Tensor, Result};

fn main() -> Result<()> {
    println!("LibTorch Rust - Basic Tensor Operations");
    
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let tensor = Tensor::new(data, vec![2, 3])?;
    
    println!("Created tensor with shape: {:?}", tensor.shape());
    tensor.print();
    
    let zeros = Tensor::zeros(vec![2, 2])?;
    println!("\nZeros tensor:");
    zeros.print();
    
    let ones = Tensor::ones(vec![3, 3])?;
    println!("\nOnes tensor:");
    ones.print();
    
    let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2])?;
    let b = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2])?;
    
    println!("\nTensor A:");
    a.print();
    println!("Tensor B:");
    b.print();
    
    let sum = a.add(&b)?;
    println!("A + B:");
    sum.print();
    
    let product = a.mul(&b)?;
    println!("A * B (element-wise):");
    product.print();
    
    let matmul_result = a.matmul(&b)?;
    println!("A @ B (matrix multiplication):");
    matmul_result.print();
    
    Ok(())
}