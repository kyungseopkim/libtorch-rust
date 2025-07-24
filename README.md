# LibTorch Rust

Rust bindings for the LibTorch C++ API, providing seamless integration of PyTorch's tensor operations and neural network capabilities in Rust applications.

## Features

- **Rust bindings for LibTorch C++ tensor operations**
- **Neural network module support through C++ API wrapper**
- **Automatic differentiation (autograd) integration**
- **CUDA/GPU acceleration support via LibTorch**
- **Multi-threaded tensor operations**
- **Custom operator registration and execution**
- **Advanced optimization algorithms (Adam, SGD, etc.)**
- **Model serialization/deserialization (save/load PyTorch models)**
- **Quantization and model pruning capabilities**
- **Mixed precision training support**
- **Memory-efficient tensor management**
- **Cross-platform compatibility (Linux, macOS, Windows)**

## Prerequisites

Before building this project, you need to have LibTorch installed. Download LibTorch from the official PyTorch website and extract it to a directory (e.g., `../libtorch`).

Set the `LIBTORCH_PATH` environment variable to point to your LibTorch installation:

```bash
export LIBTORCH_PATH=/path/to/libtorch
```

## Building

```bash
cargo build --release
```

For CPU-only builds:

```bash
cargo build --release --no-default-features --features cpu-only
```

## Usage

### Basic Tensor Operations

```rust
use libtorch_rust::{Tensor, Result};

fn main() -> Result<()> {
    // Create a tensor
    let data = vec![1.0, 2.0, 3.0, 4.0];
    let tensor = Tensor::new(data, vec![2, 2])?;
    
    // Basic operations
    let zeros = Tensor::zeros(vec![2, 2])?;
    let ones = Tensor::ones(vec![2, 2])?;
    
    // Arithmetic operations
    let sum = tensor.add(&ones)?;
    let product = tensor.mul(&ones)?;
    let matmul_result = tensor.matmul(&ones)?;
    
    Ok(())
}
```

### Neural Networks

```rust
use libtorch_rust::{Tensor, nn::{Linear, Module}, Result};

fn main() -> Result<()> {
    let input = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4])?;
    
    // Create layers
    let linear1 = Linear::new(4, 8)?;
    let linear2 = Linear::new(8, 2)?;
    
    // Forward pass
    let hidden = linear1.forward(&input)?;
    let relu_output = hidden.clamp_min(0.0)?; // ReLU activation
    let output = linear2.forward(&relu_output)?;
    
    Ok(())
}
```

### Automatic Differentiation

```rust
use libtorch_rust::{Tensor, Result};

fn main() -> Result<()> {
    let x = Tensor::new(vec![2.0, 3.0], vec![2, 1])?
        .requires_grad(true)?;
    
    let y = x.pow(2.0)?;
    let z = y.mean()?;
    
    z.backward()?;
    
    if let Some(grad) = x.grad()? {
        println!("Gradient: {:?}", grad);
    }
    
    Ok(())
}
```

### CUDA Support

```rust
use libtorch_rust::{Tensor, cuda, Result};

fn main() -> Result<()> {
    if cuda::is_available() {
        let tensor = Tensor::ones(vec![1000, 1000])?;
        let gpu_tensor = tensor.cuda()?;
        
        // Perform operations on GPU
        let result = gpu_tensor.matmul(&gpu_tensor)?;
        
        // Move back to CPU
        let cpu_result = result.cpu()?;
    }
    
    Ok(())
}
```

## Examples

Run the provided examples:

```bash
# Basic tensor operations
cargo run --example basic_tensor

# Neural network example
cargo run --example neural_network

# Autograd example  
cargo run --example autograd
```

## Testing

```bash
cargo test
```

## License

This project is licensed under either of

- Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.