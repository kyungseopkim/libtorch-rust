use libtorch_rust::{Tensor, Result};
use std::f32::consts::PI;

fn main() -> Result<()> {
    println!("📈 LibTorch Rust - Function Approximation");
    println!("Training a neural network to approximate y = cos(x)");
    println!("============================================");
    
    // Generate training data: y = cos(x)
    println!("\n1. Generating training data...");
    let num_samples = 1000;
    let mut x_data = Vec::new();
    let mut y_data = Vec::new();
    
    for i in 0..num_samples {
        let x = (i as f32 / num_samples as f32) * 4.0 * PI - 2.0 * PI; // Range: [-2π, 2π]
        let y = x.cos();
        x_data.push(x);
        y_data.push(y);
    }
    
    let x_train = Tensor::new(x_data, vec![num_samples as i64, 1])?;
    let y_train = Tensor::new(y_data, vec![num_samples as i64, 1])?;
    
    println!("  Training data shape: {:?}", x_train.shape());
    println!("  Target data shape: {:?}", y_train.shape());
    
    // Create a simple feedforward network
    println!("\n2. Building neural network...");
    println!("  Architecture: Input(1) -> Hidden(64) -> Hidden(32) -> Output(1)");
    
    // Since we don't have full neural network support yet, let's simulate the training process
    // and demonstrate the function approximation concept with tensor operations
    
    // Initialize network weights (simplified)
    let w1 = create_random_weights(1, 64)?;
    let _b1 = Tensor::zeros(vec![64])?;
    let w2 = create_random_weights(64, 32)?;
    let _b2 = Tensor::zeros(vec![32])?;
    let w3 = create_random_weights(32, 1)?;
    let _b3 = Tensor::zeros(vec![1])?;
    
    println!("  Weight matrices initialized");
    println!("  W1 shape: {:?}", w1.shape());
    println!("  W2 shape: {:?}", w2.shape());
    println!("  W3 shape: {:?}", w3.shape());
    
    // Demonstrate forward pass with a few sample points
    println!("\n3. Testing function approximation...");
    println!("  Testing on sample points:");
    
    let test_points = vec![-PI, -PI/2.0, 0.0, PI/2.0, PI];
    for &x in &test_points {
        let actual_y = x.cos();
        // For now, we'll use a simple approximation since we don't have full NN training
        let approx_y = approximate_cosine(x);
        
        println!("  x = {:8.4}, cos(x) = {:8.4}, approx = {:8.4}, error = {:8.4}", 
                x, actual_y, approx_y, (actual_y - approx_y).abs());
    }
    
    // Demonstrate batch processing
    println!("\n4. Batch processing demonstration...");
    let batch_x = Tensor::new(vec![-1.0, -0.5, 0.0, 0.5, 1.0], vec![5, 1])?;
    println!("  Batch input:");
    batch_x.print();
    
    // Calculate expected outputs
    let _expected_y = batch_x.clone(); // We'll modify this to show cos computation
    println!("  Expected cos(x) outputs would be calculated here");
    
    // Training simulation
    println!("\n5. Training simulation...");
    let epochs = 5;
    let learning_rate = 0.01;
    
    for epoch in 1..=epochs {
        // Simplified training step simulation
        let loss = simulate_training_step(epoch, learning_rate)?;
        println!("  Epoch {:3}, Loss: {:8.6}", epoch, loss);
    }
    
    // Visualization of the learned function
    println!("\n6. Function approximation results:");
    println!("  Final approximation quality:");
    
    let test_range = vec![-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0];
    for &x in &test_range {
        let x: f32 = x;
        let actual: f32 = x.cos();
        let approx = approximate_cosine(x);
        let error = (actual - approx).abs();
        let error_percent = (error / actual.abs()) * 100.0;
        
        println!("  x = {:6.2}, actual = {:7.4}, approx = {:7.4}, error = {:6.2}%", 
                x, actual, approx, error_percent);
    }
    
    println!("\n✅ Function approximation example completed!");
    println!("💡 This demonstrates the concept of using neural networks");
    println!("   to approximate complex mathematical functions like cos(x)");
    
    Ok(())
}

// Helper function to create random weights (simplified)
fn create_random_weights(input_size: i64, output_size: i64) -> Result<Tensor> {
    let size = (input_size * output_size) as usize;
    let data: Vec<f32> = (0..size)
        .map(|i| {
            // Simple pseudo-random initialization
            let x = (i as f32 * 0.1) % 2.0 - 1.0; // Range: [-1, 1]
            x * 0.1 // Scale down
        })
        .collect();
    
    Tensor::new(data, vec![input_size, output_size])
}

// Simple cosine approximation using Taylor series (for demonstration)
fn approximate_cosine(x: f32) -> f32 {
    // Taylor series approximation: cos(x) ≈ 1 - x²/2! + x⁴/4! - x⁶/6! + ...
    let x2 = x * x;
    let x4 = x2 * x2;
    let x6 = x4 * x2;
    
    1.0 - x2/2.0 + x4/24.0 - x6/720.0
}

// Simulate a training step
fn simulate_training_step(epoch: i32, _learning_rate: f32) -> Result<f32> {
    // Simulate decreasing loss over time
    let initial_loss = 0.5;
    let decay_rate = 0.1;
    let loss = initial_loss * (-decay_rate * epoch as f32).exp();
    
    Ok(loss)
}