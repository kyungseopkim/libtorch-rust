use libtorch_rust::{Tensor, Result};

fn main() -> Result<()> {
    println!("🎯 LibTorch Rust - Simple Optimization");
    println!("Demonstrating gradient descent optimization");
    println!("=========================================");
    
    // Problem: Find minimum of f(x) = (x - 3)² + 2
    // The minimum is at x = 3, with f(3) = 2
    println!("\n1. Optimization Problem:");
    println!("  Minimize: f(x) = (x - 3)² + 2");
    println!("  True minimum: x = 3, f(3) = 2");
    
    // Initialize parameter
    let mut x = Tensor::new(vec![0.0], vec![1])?; // Start at x = 0
    let learning_rate = 0.1;
    let num_iterations = 50;
    
    println!("\n2. Initial conditions:");
    println!("  Starting point: x = 0.0");
    println!("  Learning rate: {}", learning_rate);
    println!("  Iterations: {}", num_iterations);
    
    // Optimization loop
    println!("\n3. Optimization progress:");
    println!("  Iter |    x    |  f(x)   | gradient");
    println!("  -----|---------|---------|----------");
    
    for iter in 0..num_iterations {
        // Get current x value
        let x_val = get_tensor_value(&x)?;
        
        // Compute function value: f(x) = (x - 3)² + 2
        let f_val = (x_val - 3.0).powi(2) + 2.0;
        
        // Compute gradient: f'(x) = 2(x - 3)
        let gradient = 2.0 * (x_val - 3.0);
        
        // Print progress every 5 iterations
        if iter % 5 == 0 || iter < 10 {
            println!("  {:4} | {:7.4} | {:7.4} | {:8.4}", 
                    iter, x_val, f_val, gradient);
        }
        
        // Update parameter: x = x - learning_rate * gradient
        let new_x_val = x_val - learning_rate * gradient;
        x = Tensor::new(vec![new_x_val], vec![1])?;
        
        // Check convergence
        if gradient.abs() < 1e-6 {
            println!("  Converged at iteration {}", iter);
            break;
        }
    }
    
    let final_x = get_tensor_value(&x)?;
    let final_f = (final_x - 3.0).powi(2) + 2.0;
    
    println!("\n4. Final results:");
    println!("  Final x: {:8.6}", final_x);
    println!("  Final f(x): {:8.6}", final_f);
    println!("  Error from true minimum: {:8.6}", (final_x - 3.0).abs());
    
    // Multi-dimensional optimization example
    println!("\n5. Multi-dimensional optimization:");
    println!("  Minimize: f(x,y) = (x-1)² + (y-2)² + 3");
    println!("  True minimum: x=1, y=2, f(1,2)=3");
    
    let mut params = Tensor::new(vec![0.0, 0.0], vec![2])?; // Start at (0,0)
    let lr_2d = 0.1;
    let iterations_2d = 30;
    
    println!("\n  Iter |   x    |   y    |  f(x,y) | grad_x | grad_y");
    println!("  -----|--------|--------|---------|--------|--------");
    
    for iter in 0..iterations_2d {
        let param_vals = get_tensor_values(&params, 2)?;
        let x_val = param_vals[0];
        let y_val = param_vals[1];
        
        // Function value: f(x,y) = (x-1)² + (y-2)² + 3
        let f_val = (x_val - 1.0).powi(2) + (y_val - 2.0).powi(2) + 3.0;
        
        // Gradients: ∂f/∂x = 2(x-1), ∂f/∂y = 2(y-2)
        let grad_x = 2.0 * (x_val - 1.0);
        let grad_y = 2.0 * (y_val - 2.0);
        
        if iter % 5 == 0 || iter < 10 {
            println!("  {:4} | {:6.3} | {:6.3} | {:7.4} | {:6.3} | {:6.3}", 
                    iter, x_val, y_val, f_val, grad_x, grad_y);
        }
        
        // Update parameters
        let new_x = x_val - lr_2d * grad_x;
        let new_y = y_val - lr_2d * grad_y;
        params = Tensor::new(vec![new_x, new_y], vec![2])?;
        
        // Check convergence
        if grad_x.abs() < 1e-6 && grad_y.abs() < 1e-6 {
            println!("  Converged at iteration {}", iter);
            break;
        }
    }
    
    let final_params = get_tensor_values(&params, 2)?;
    let final_f_2d = (final_params[0] - 1.0).powi(2) + (final_params[1] - 2.0).powi(2) + 3.0;
    
    println!("\n6. Final 2D results:");
    println!("  Final x: {:8.6}", final_params[0]);
    println!("  Final y: {:8.6}", final_params[1]);
    println!("  Final f(x,y): {:8.6}", final_f_2d);
    println!("  Error from true minimum:");
    println!("    |x - 1|: {:8.6}", (final_params[0] - 1.0).abs());
    println!("    |y - 2|: {:8.6}", (final_params[1] - 2.0).abs());
    
    // Demonstrate different optimization algorithms
    println!("\n7. Optimization algorithms comparison:");
    demonstrate_optimization_methods()?;
    
    println!("\n✅ Simple optimization example completed!");
    println!("💡 This demonstrates the fundamentals of gradient descent");
    println!("   and how it can be used to minimize mathematical functions");
    
    Ok(())
}

// Helper function to extract scalar value from tensor
fn get_tensor_value(tensor: &Tensor) -> Result<f32> {
    let data_ptr = tensor.data_ptr();
    let value = unsafe { *data_ptr };
    Ok(value)
}

// Helper function to extract multiple values from tensor
fn get_tensor_values(tensor: &Tensor, count: usize) -> Result<Vec<f32>> {
    let data_ptr = tensor.data_ptr();
    let mut values = Vec::new();
    
    for i in 0..count {
        let value = unsafe { *data_ptr.add(i) };
        values.push(value);
    }
    
    Ok(values)
}

// Demonstrate different optimization methods
fn demonstrate_optimization_methods() -> Result<()> {
    println!("  Comparing optimization methods on f(x) = x² - 4x + 7");
    println!("  True minimum: x = 2, f(2) = 3");
    
    // Method 1: Standard Gradient Descent
    let result1 = optimize_standard_gd()?;
    println!("  Standard GD:     x = {:6.3}, iterations = {}", result1.0, result1.1);
    
    // Method 2: Gradient Descent with Momentum (simulated)
    let result2 = optimize_with_momentum()?;
    println!("  GD with Momentum: x = {:6.3}, iterations = {}", result2.0, result2.1);
    
    // Method 3: Adaptive Learning Rate
    let result3 = optimize_adaptive_lr()?;
    println!("  Adaptive LR:     x = {:6.3}, iterations = {}", result3.0, result3.1);
    
    Ok(())
}

fn optimize_standard_gd() -> Result<(f32, i32)> {
    let mut x = 0.0;
    let lr = 0.1;
    
    for iter in 0..100 {
        let gradient: f32 = 2.0 * x - 4.0; // f'(x) = 2x - 4
        x = x - lr * gradient;
        
        if gradient.abs() < 1e-6 {
            return Ok((x, iter));
        }
    }
    
    Ok((x, 100))
}

fn optimize_with_momentum() -> Result<(f32, i32)> {
    let mut x = 0.0;
    let mut velocity = 0.0;
    let lr = 0.1;
    let momentum = 0.9;
    
    for iter in 0..100 {
        let gradient: f32 = 2.0 * x - 4.0;
        velocity = momentum * velocity + lr * gradient;
        x = x - velocity;
        
        if gradient.abs() < 1e-6 {
            return Ok((x, iter));
        }
    }
    
    Ok((x, 100))
}

fn optimize_adaptive_lr() -> Result<(f32, i32)> {
    let mut x = 0.0;
    let mut lr = 0.5;
    
    for iter in 0..100 {
        let gradient: f32 = 2.0 * x - 4.0;
        
        // Adaptive learning rate: reduce if gradient is small
        if gradient.abs() < 0.5 {
            lr = 0.1;
        }
        
        x = x - lr * gradient;
        
        if gradient.abs() < 1e-6 {
            return Ok((x, iter));
        }
    }
    
    Ok((x, 100))
}