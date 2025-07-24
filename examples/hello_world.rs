use libtorch_rust::{Tensor, Result};

fn main() -> Result<()> {
    println!("🌍 LibTorch Rust - Hello World!");
    println!("==================================");
    
    // Create a simple tensor
    println!("\n1. Creating a simple tensor:");
    let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2])?;
    println!("Created 2x2 tensor:");
    tensor.print();
    
    // Check tensor properties
    println!("\n2. Tensor properties:");
    println!("  Shape: {:?}", tensor.shape());
    println!("  Data type: {:?}", tensor.dtype());
    println!("  Number of elements: {}", tensor.numel());
    
    // Create different types of tensors
    println!("\n3. Creating different tensor types:");
    
    let zeros = Tensor::zeros(vec![3, 3])?;
    println!("3x3 Zero tensor:");
    zeros.print();
    
    let ones = Tensor::ones(vec![2, 4])?;
    println!("2x4 Ones tensor:");
    ones.print();
    
    // Basic tensor operations
    println!("\n4. Basic tensor operations:");
    let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2])?;
    let b = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2])?;
    
    println!("Tensor A:");
    a.print();
    println!("Tensor B:");
    b.print();
    
    let sum = a.add(&b)?;
    println!("A + B:");
    sum.print();
    
    let product = a.mul(&b)?;
    println!("A * B (element-wise):");
    product.print();
    
    let matmul = a.matmul(&b)?;
    println!("A @ B (matrix multiplication):");
    matmul.print();
    
    // Vector operations
    println!("\n5. Vector operations:");
    let vec1 = Tensor::new(vec![1.0, 2.0, 3.0], vec![3])?;
    let vec2 = Tensor::new(vec![4.0, 5.0, 6.0], vec![3])?;
    
    println!("Vector 1:");
    vec1.print();
    println!("Vector 2:");
    vec2.print();
    
    let vec_sum = vec1.add(&vec2)?;
    println!("Vector sum:");
    vec_sum.print();
    
    // Reshape operations
    println!("\n6. Tensor reshaping:");
    let original = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3])?;
    println!("Original tensor (2x3):");
    original.print();
    
    let reshaped = original.reshape(vec![3, 2])?;
    println!("Reshaped tensor (3x2):");
    reshaped.print();
    
    println!("\n✅ Hello World example completed successfully!");
    println!("LibTorch Rust bindings are working properly! 🦀");
    
    Ok(())
}