use libtorch_rust::{Tensor, nn::{Linear, Module}, Result};

fn main() -> Result<()> {
    println!("LibTorch Rust - Neural Network Example");
    
    let input = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4])?;
    println!("Input tensor:");
    input.print();
    
    let linear1 = Linear::new(4, 8)?;
    println!("\nCreated Linear layer: {} -> {}", linear1.in_features(), linear1.out_features());
    
    let hidden = linear1.forward(&input)?;
    println!("After first linear layer:");
    hidden.print();
    
    let relu_output = hidden.clamp_min(0.0)?;
    println!("After ReLU activation:");
    relu_output.print();
    
    let linear2 = Linear::new(8, 2)?;
    let output = linear2.forward(&relu_output)?;
    println!("Final output:");
    output.print();
    
    let softmax_output = output.softmax(-1)?;
    println!("After softmax:");
    softmax_output.print();
    
    Ok(())
}