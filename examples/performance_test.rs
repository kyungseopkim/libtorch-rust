use libtorch_rust::{Tensor, Result};
use std::time::Instant;

fn main() -> Result<()> {
    println!("⚡ LibTorch Rust - Performance Test ⚡");
    println!("=====================================");
    
    // Small matrix test
    println!("\n🚀 Small Matrix Operations (100x100):");
    let start = Instant::now();
    
    let a = create_test_matrix(100, 100)?;
    let b = create_test_matrix(100, 100)?;
    let creation_time = start.elapsed();
    
    let start = Instant::now();
    let result = a.matmul(&b)?;
    let matmul_time = start.elapsed();
    
    println!("  ✅ Creation: {:?}", creation_time);
    println!("  ✅ Matrix Multiplication: {:?}", matmul_time);
    println!("  📊 Result shape: {:?}", result.shape());
    
    // Medium matrix test
    println!("\n🚀 Medium Matrix Operations (500x500):");
    let start = Instant::now();
    
    let a = create_test_matrix(500, 500)?;
    let b = create_test_matrix(500, 500)?;
    let creation_time = start.elapsed();
    
    let start = Instant::now();
    let result = a.matmul(&b)?;
    let matmul_time = start.elapsed();
    
    println!("  ✅ Creation: {:?}", creation_time);
    println!("  ✅ Matrix Multiplication: {:?}", matmul_time);
    println!("  📊 Result shape: {:?}", result.shape());
    
    // Element-wise operations test
    println!("\n🧮 Element-wise Operations (1000x1000):");
    let start = Instant::now();
    
    let a = create_test_matrix(1000, 1000)?;
    let b = create_test_matrix(1000, 1000)?;
    let creation_time = start.elapsed();
    
    let start = Instant::now();
    let _sum = a.add(&b)?;
    let add_time = start.elapsed();
    
    let start = Instant::now();  
    let product = a.mul(&b)?;
    let mul_time = start.elapsed();
    
    println!("  ✅ Creation: {:?}", creation_time);
    println!("  ✅ Addition: {:?}", add_time);
    println!("  ✅ Element-wise Multiplication: {:?}", mul_time);
    println!("  📊 Elements processed: {}", product.numel());
    
    println!("\n💯 Performance test completed!");
    
    Ok(())
}

fn create_test_matrix(rows: i64, cols: i64) -> Result<Tensor> {
    let size = (rows * cols) as usize;
    let data: Vec<f32> = (0..size)
        .map(|i| (i as f32 * 0.01) % 10.0)
        .collect();
    
    Tensor::new(data, vec![rows, cols])
}