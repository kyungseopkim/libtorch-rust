use libtorch_rust::{Tensor, Result};

fn main() -> Result<()> {
    println!("🦀 LibTorch Rust - Simple Demo 🦀");
    println!("=====================================");
    
    // Create basic tensors
    println!("\n📦 Creating tensors...");
    let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2])?;
    println!("Tensor A (2x2):");
    a.print();
    
    let b = Tensor::new(vec![0.5, 1.5, 2.5, 3.5], vec![2, 2])?;
    println!("Tensor B (2x2):");
    b.print();
    
    // Basic arithmetic
    println!("\n🧮 Basic arithmetic operations...");
    let sum = a.add(&b)?;
    println!("A + B =");
    sum.print();
    
    let product = a.mul(&b)?;
    println!("A * B (element-wise) =");
    product.print();
    
    // Matrix multiplication
    println!("\n🔢 Matrix multiplication...");
    let matmul = a.matmul(&b)?;
    println!("A @ B =");
    matmul.print();
    
    // Create special tensors
    println!("\n✨ Special tensor creation...");
    let zeros = Tensor::zeros(vec![3, 2])?;
    println!("Zeros (3x2):");
    zeros.print();
    
    let ones = Tensor::ones(vec![2, 3])?;
    println!("Ones (2x3):");
    ones.print();
    
    // Larger example
    println!("\n🚀 Larger tensor operations...");
    let large_a = Tensor::new(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], 
        vec![3, 3]
    )?;
    let large_b = Tensor::new(
        vec![9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0], 
        vec![3, 3]
    )?;
    
    println!("Large Matrix A (3x3):");
    large_a.print();
    println!("Large Matrix B (3x3):");
    large_b.print();
    
    let large_result = large_a.matmul(&large_b)?;
    println!("A @ B (3x3 result):");
    large_result.print();
    
    println!("\n✅ All operations completed successfully!");
    println!("📊 Tensor info:");
    println!("  - Shape: {:?}", large_result.shape());
    println!("  - Data type: {:?}", large_result.dtype());
    println!("  - Number of elements: {}", large_result.numel());
    
    Ok(())
}