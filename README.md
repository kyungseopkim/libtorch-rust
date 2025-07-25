# LibTorch Rust

Rust bindings for the LibTorch C++ API, providing tensor operations and basic neural network capabilities for Rust applications.

## Current Status

This project provides working Rust bindings for core LibTorch tensor operations. The implementation includes:

✅ **Implemented Features:**
- Basic tensor creation and manipulation
- Element-wise operations (add, multiply)
- Matrix multiplication (matmul)
- Tensor reshaping
- Memory-efficient tensor management with proper Drop implementations
- CPU-only operations (default)
- Optional CUDA support
- Cross-platform compatibility (Linux, macOS, Windows)

🚧 **Planned Features:**
- Neural network modules (Linear layers, etc.)
- Automatic differentiation (autograd)
- Advanced optimization algorithms
- Model serialization/deserialization
- Quantization and model pruning

## Prerequisites

Before building this project, you need LibTorch installed:

1. Download LibTorch from the [official PyTorch website](https://pytorch.org/get-started/locally/)
2. Extract it to a directory (e.g., `../libtorch`)
3. Set the `LIBTORCH_PATH` environment variable (optional, defaults to `../libtorch`):

```bash
export LIBTORCH_PATH=/path/to/libtorch
```

## Building

### CPU-only (Default)
```bash
cargo build --release
```

### With CUDA support
```bash
cargo build --release --features cuda
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
    
    // Reshape tensor
    let reshaped = tensor.reshape(vec![1, 4])?;
    
    // Get tensor properties
    println!("Shape: {:?}", tensor.shape());
    println!("Element count: {}", tensor.numel());
    println!("Data type: {}", tensor.dtype());
    
    Ok(())
}
```

### Running with LibTorch

Since the library depends on LibTorch dynamic libraries, you need to set the library path when running:

```bash
# On macOS/Linux
DYLD_LIBRARY_PATH=../libtorch/lib cargo run

# On Linux (alternative)
LD_LIBRARY_PATH=../libtorch/lib cargo run
```

## Examples

Run the provided examples:

```bash
# Test runner with comprehensive tests
DYLD_LIBRARY_PATH=../libtorch/lib cargo run --example test_runner

# Performance benchmarks
DYLD_LIBRARY_PATH=../libtorch/lib cargo run --example performance_test
```

## Testing

### Quick Test Script
```bash
./run_tests.sh
```

### Manual Testing
```bash
# Library tests only
DYLD_LIBRARY_PATH=../libtorch/lib cargo test --lib

# All tests (library + examples)
DYLD_LIBRARY_PATH=../libtorch/lib cargo test
```

## Project Structure

```
src/
├── lib.rs              # Main library entry point
├── error.rs            # Error handling
├── tensor/mod.rs       # Tensor implementation
├── cpp/torch_wrapper.cpp # C++ wrapper functions
└── simple_test.rs      # Comprehensive test suite

examples/
├── test_runner.rs      # Full test suite runner
└── performance_test.rs # Performance benchmarks

examples_backup/        # Advanced examples (work in progress)
├── neural_network.rs   # Neural network layers
├── autograd.rs         # Automatic differentiation
└── ...                 # Other advanced features
```

## API Reference

### Tensor Creation
- `Tensor::new(data: Vec<f32>, shape: Vec<i64>)` - Create tensor from data
- `Tensor::zeros(shape: Vec<i64>)` - Create zero-filled tensor
- `Tensor::ones(shape: Vec<i64>)` - Create ones-filled tensor

### Tensor Operations
- `tensor.add(&other)` - Element-wise addition
- `tensor.mul(&other)` - Element-wise multiplication
- `tensor.matmul(&other)` - Matrix multiplication
- `tensor.reshape(new_shape: Vec<i64>)` - Reshape tensor

### Tensor Properties
- `tensor.shape()` - Get tensor dimensions
- `tensor.numel()` - Get total number of elements
- `tensor.dtype()` - Get data type

## Features

The project supports the following cargo features:

- `default = []` - CPU-only build (default)
- `cuda` - Enable CUDA/GPU support
- `cpu-only` - Explicitly CPU-only (same as default)

## Development

### Build Configuration

The build process uses:
- `bindgen` for generating Rust FFI bindings from C++ headers
- `cc` crate for compiling the C++ wrapper code
- Automatic LibTorch library linking

### Contributing

1. Ensure all tests pass: `./run_tests.sh`
2. Check that examples compile: `cargo check --examples`
3. Follow the existing code patterns for FFI and memory management

## License

This project is licensed under either of

- Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. Make sure to:

- Add tests for new functionality
- Update documentation
- Follow Rust best practices for FFI and memory safety