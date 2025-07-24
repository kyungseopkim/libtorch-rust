use libtorch_rust::{Tensor, Result};

fn main() -> Result<()> {
    println!("LibTorch Rust - Autograd Example");
    
    let x = Tensor::new(vec![2.0, 3.0], vec![2, 1])?
        .requires_grad(true)?;
    
    println!("Input tensor x (requires_grad=true):");
    x.print();
    
    let y = x.pow(2.0)?;
    println!("y = x^2:");
    y.print();
    
    let z = y.mean()?;
    println!("z = mean(y):");
    z.print();
    
    z.backward()?;
    println!("After backward pass");
    
    if let Some(grad) = x.grad()? {
        println!("Gradient of x:");
        grad.print();
    } else {
        println!("No gradient found for x");
    }
    
    Ok(())
}