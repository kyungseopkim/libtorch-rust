use libtorch_rust::{Tensor, Result};
use std::f32::consts::PI;

fn main() -> Result<()> {
    println!("🔢 LibTorch Rust - Custom Loss Functions");
    println!("Implementing and comparing different loss functions");
    println!("===============================================");
    
    // Generate synthetic data for regression
    println!("\n1. Generating synthetic regression data:");
    let (x_data, y_true, y_pred) = generate_regression_data(100)?;
    println!("  Data points: {}", x_data.len());
    println!("  Feature range: [{:.2}, {:.2}]", 
             x_data.iter().fold(f32::INFINITY, |a, &b| a.min(b)),
             x_data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)));
    
    // Show first few samples
    println!("  First 5 samples:");
    for i in 0..5 {
        println!("    x = {:6.3}, y_true = {:6.3}, y_pred = {:6.3}", 
                x_data[i], y_true[i], y_pred[i]);
    }
    
    // Convert to tensors
    let y_true_tensor = Tensor::new(y_true.clone(), vec![y_true.len() as i64])?;
    let y_pred_tensor = Tensor::new(y_pred.clone(), vec![y_pred.len() as i64])?;
    
    // 2. Regression Loss Functions
    println!("\n2. Regression loss functions:");
    
    // Mean Squared Error (MSE)
    let mse_loss = mean_squared_error(&y_true, &y_pred)?;
    println!("  Mean Squared Error (MSE): {:.6}", mse_loss);
    
    // Mean Absolute Error (MAE)
    let mae_loss = mean_absolute_error(&y_true, &y_pred)?;
    println!("  Mean Absolute Error (MAE): {:.6}", mae_loss);
    
    // Root Mean Squared Error (RMSE)
    let rmse_loss = root_mean_squared_error(&y_true, &y_pred)?;
    println!("  Root Mean Squared Error (RMSE): {:.6}", rmse_loss);
    
    // Huber Loss (robust to outliers)
    let huber_loss = huber_loss(&y_true, &y_pred, 1.0)?;
    println!("  Huber Loss (δ=1.0): {:.6}", huber_loss);
    
    // Log-Cosh Loss
    let log_cosh_loss = log_cosh_loss(&y_true, &y_pred)?;
    println!("  Log-Cosh Loss: {:.6}", log_cosh_loss);
    
    // 3. Classification Loss Functions
    println!("\n3. Classification loss functions:");
    
    // Generate binary classification data
    let (class_true, class_pred_logits, class_pred_probs) = generate_classification_data(100)?;
    
    println!("  Binary classification samples:");
    for i in 0..5 {
        println!("    true = {}, logit = {:6.3}, prob = {:6.3}", 
                class_true[i], class_pred_logits[i], class_pred_probs[i]);
    }
    
    // Binary Cross-Entropy Loss
    let bce_loss = binary_cross_entropy_loss(&class_true, &class_pred_probs)?;
    println!("  Binary Cross-Entropy: {:.6}", bce_loss);
    
    // Logistic Loss
    let logistic_loss = logistic_loss(&class_true, &class_pred_logits)?;
    println!("  Logistic Loss: {:.6}", logistic_loss);
    
    // Hinge Loss (SVM)
    let hinge_loss = hinge_loss(&class_true, &class_pred_logits)?;
    println!("  Hinge Loss: {:.6}", hinge_loss);
    
    // 4. Custom Loss Functions
    println!("\n4. Custom loss functions:");
    
    // Custom weighted MSE
    let weights = vec![1.0; y_true.len()]; // Equal weights for now
    let weighted_mse = weighted_mse_loss(&y_true, &y_pred, &weights)?;
    println!("  Weighted MSE: {:.6}", weighted_mse);
    
    // Custom asymmetric loss (penalizes underestimation more)
    let asymmetric_loss = asymmetric_loss(&y_true, &y_pred, 0.3, 0.7)?;
    println!("  Asymmetric Loss (α=0.3, β=0.7): {:.6}", asymmetric_loss);
    
    // Custom quantile loss
    let quantile_loss_50 = quantile_loss(&y_true, &y_pred, 0.5)?; // Median
    let quantile_loss_90 = quantile_loss(&y_true, &y_pred, 0.9)?; // 90th percentile
    println!("  Quantile Loss (τ=0.5): {:.6}", quantile_loss_50);
    println!("  Quantile Loss (τ=0.9): {:.6}", quantile_loss_90);
    
    // 5. Loss Function Comparisons
    println!("\n5. Loss function behavior analysis:");
    analyze_loss_behavior()?;
    
    // 6. Gradient Analysis
    println!("\n6. Loss function gradients:");
    analyze_gradients(&y_true, &y_pred)?;
    
    // 7. Robust Loss Functions
    println!("\n7. Robust loss functions (with outliers):");
    let (y_true_outliers, y_pred_outliers) = add_outliers(&y_true, &y_pred)?;
    
    let mse_outliers = mean_squared_error(&y_true_outliers, &y_pred_outliers)?;
    let mae_outliers = mean_absolute_error(&y_true_outliers, &y_pred_outliers)?;
    let huber_outliers = huber_loss(&y_true_outliers, &y_pred_outliers, 1.0)?;
    
    println!("  With outliers:");
    println!("    MSE: {:.6} (sensitive to outliers)", mse_outliers);
    println!("    MAE: {:.6} (robust to outliers)", mae_outliers);
    println!("    Huber: {:.6} (balanced robustness)", huber_outliers);
    
    // 8. Multi-class Classification
    println!("\n8. Multi-class classification losses:");
    let (true_classes, pred_logits) = generate_multiclass_data(50, 3)?;
    
    let categorical_ce = categorical_cross_entropy(&true_classes, &pred_logits, 3)?;
    let sparse_ce = sparse_categorical_cross_entropy(&true_classes, &pred_logits, 3)?;
    
    println!("  Categorical Cross-Entropy: {:.6}", categorical_ce);
    println!("  Sparse Categorical Cross-Entropy: {:.6}", sparse_ce);
    
    // 9. Custom Training Loop Simulation
    println!("\n9. Training loop with different losses:");
    simulate_training_with_different_losses()?;
    
    // 10. Loss Function Recommendations
    println!("\n10. Loss function selection guide:");
    print_loss_function_guide();
    
    println!("\n✅ Custom loss functions example completed!");
    println!("💡 This demonstrates:");
    println!("   - Common regression and classification losses");  
    println!("   - Custom loss function implementations");
    println!("   - Robustness properties of different losses");
    println!("   - Gradient behavior analysis");
    
    Ok(())
}

// Data generation functions
fn generate_regression_data(n_samples: usize) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    let mut x_data = Vec::new();
    let mut y_true = Vec::new();
    let mut y_pred = Vec::new();
    
    for i in 0..n_samples {
        let x = (i as f32 / n_samples as f32) * 10.0 - 5.0; // Range: [-5, 5]
        let y_true_val = 2.0 * x + 1.0 + (x * 0.1).sin() * 2.0; // True function
        
        // Add prediction noise
        let noise = ((i as f32 * 0.123) % 2.0) - 1.0; // Simple noise [-1, 1]
        let y_pred_val = y_true_val + noise * 0.5;
        
        x_data.push(x);
        y_true.push(y_true_val);
        y_pred.push(y_pred_val);
    }
    
    Ok((x_data, y_true, y_pred))
}

fn generate_classification_data(n_samples: usize) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    let mut y_true = Vec::new();
    let mut y_pred_logits = Vec::new();
    let mut y_pred_probs = Vec::new();
    
    for i in 0..n_samples {
        // Generate binary labels
        let true_label = if i % 3 == 0 { 1.0 } else { 0.0 };
        
        // Generate logits (pre-sigmoid values)
        let logit = if true_label == 1.0 {
            1.0 + ((i as f32 * 0.1) % 2.0) // Positive logits for class 1
        } else {
            -1.0 + ((i as f32 * 0.1) % 2.0) // Negative logits for class 0
        };
        
        // Convert to probability
        let prob = 1.0 / (1.0 + (-logit).exp()); // Sigmoid function
        
        y_true.push(true_label);
        y_pred_logits.push(logit);
        y_pred_probs.push(prob);
    }
    
    Ok((y_true, y_pred_logits, y_pred_probs))
}

fn generate_multiclass_data(n_samples: usize, n_classes: usize) -> Result<(Vec<usize>, Vec<Vec<f32>>)> {
    let mut true_classes = Vec::new();
    let mut pred_logits = Vec::new();
    
    for i in 0..n_samples {
        let true_class = i % n_classes;
        true_classes.push(true_class);
        
        // Generate logits for each class
        let mut logits = Vec::new();
        for j in 0..n_classes {
            let base_logit = if j == true_class { 2.0 } else { 0.0 };
            let noise = ((i + j) as f32 * 0.1) % 2.0 - 1.0;
            logits.push(base_logit + noise * 0.5);
        }
        
        pred_logits.push(logits);
    }
    
    Ok((true_classes, pred_logits))
}

// Regression Loss Functions
fn mean_squared_error(y_true: &[f32], y_pred: &[f32]) -> Result<f32> {
    if y_true.len() != y_pred.len() {
        return Err(libtorch_rust::TorchError::TensorError("Length mismatch".to_string()));
    }
    
    let mse = y_true.iter()
        .zip(y_pred.iter())
        .map(|(&t, &p)| (t - p).powi(2))
        .sum::<f32>() / y_true.len() as f32;
    
    Ok(mse)
}

fn mean_absolute_error(y_true: &[f32], y_pred: &[f32]) -> Result<f32> {
    if y_true.len() != y_pred.len() {
        return Err(libtorch_rust::TorchError::TensorError("Length mismatch".to_string()));
    }
    
    let mae = y_true.iter()
        .zip(y_pred.iter())
        .map(|(&t, &p)| (t - p).abs())
        .sum::<f32>() / y_true.len() as f32;
    
    Ok(mae)
}

fn root_mean_squared_error(y_true: &[f32], y_pred: &[f32]) -> Result<f32> {
    let mse = mean_squared_error(y_true, y_pred)?;
    Ok(mse.sqrt())
}

fn huber_loss(y_true: &[f32], y_pred: &[f32], delta: f32) -> Result<f32> {
    if y_true.len() != y_pred.len() {
        return Err(libtorch_rust::TorchError::TensorError("Length mismatch".to_string()));
    }
    
    let huber = y_true.iter()
        .zip(y_pred.iter())
        .map(|(&t, &p)| {
            let diff = (t - p).abs();
            if diff <= delta {
                0.5 * diff.powi(2)
            } else {
                delta * (diff - 0.5 * delta)
            }
        })
        .sum::<f32>() / y_true.len() as f32;
    
    Ok(huber)
}

fn log_cosh_loss(y_true: &[f32], y_pred: &[f32]) -> Result<f32> {
    if y_true.len() != y_pred.len() {
        return Err(libtorch_rust::TorchError::TensorError("Length mismatch".to_string()));
    }
    
    let log_cosh = y_true.iter()
        .zip(y_pred.iter())
        .map(|(&t, &p)| {
            let diff = t - p;
            (diff.cosh()).ln()
        })
        .sum::<f32>() / y_true.len() as f32;
    
    Ok(log_cosh)
}

// Classification Loss Functions
fn binary_cross_entropy_loss(y_true: &[f32], y_pred_probs: &[f32]) -> Result<f32> {
    if y_true.len() != y_pred_probs.len() {
        return Err(libtorch_rust::TorchError::TensorError("Length mismatch".to_string()));
    }
    
    let bce = y_true.iter()
        .zip(y_pred_probs.iter())
        .map(|(&t, &p)| {
            let p_clamped = p.max(1e-7).min(1.0 - 1e-7); // Avoid log(0)
            -(t * p_clamped.ln() + (1.0 - t) * (1.0 - p_clamped).ln())
        })
        .sum::<f32>() / y_true.len() as f32;
    
    Ok(bce)
}

fn logistic_loss(y_true: &[f32], y_pred_logits: &[f32]) -> Result<f32> {
    if y_true.len() != y_pred_logits.len() {
        return Err(libtorch_rust::TorchError::TensorError("Length mismatch".to_string()));
    }
    
    let logistic = y_true.iter()
        .zip(y_pred_logits.iter())
        .map(|(&t, &logit)| {
            let y = if t == 1.0 { 1.0 } else { -1.0 }; // Convert to {-1, 1}
            (1.0 + (-y * logit).exp()).ln()
        })
        .sum::<f32>() / y_true.len() as f32;
    
    Ok(logistic)
}

fn hinge_loss(y_true: &[f32], y_pred_logits: &[f32]) -> Result<f32> {
    if y_true.len() != y_pred_logits.len() {
        return Err(libtorch_rust::TorchError::TensorError("Length mismatch".to_string()));
    }
    
    let hinge = y_true.iter()
        .zip(y_pred_logits.iter())
        .map(|(&t, &logit)| {
            let y = if t == 1.0 { 1.0 } else { -1.0 }; // Convert to {-1, 1}
            (1.0 - y * logit).max(0.0)
        })
        .sum::<f32>() / y_true.len() as f32;
    
    Ok(hinge)
}

// Custom Loss Functions
fn weighted_mse_loss(y_true: &[f32], y_pred: &[f32], weights: &[f32]) -> Result<f32> {
    if y_true.len() != y_pred.len() || y_true.len() != weights.len() {
        return Err(libtorch_rust::TorchError::TensorError("Length mismatch".to_string()));
    }
    
    let weighted_mse = y_true.iter()
        .zip(y_pred.iter())
        .zip(weights.iter())
        .map(|((&t, &p), &w)| w * (t - p).powi(2))
        .sum::<f32>() / weights.iter().sum::<f32>();
    
    Ok(weighted_mse)
}

fn asymmetric_loss(y_true: &[f32], y_pred: &[f32], alpha: f32, beta: f32) -> Result<f32> {
    if y_true.len() != y_pred.len() {
        return Err(libtorch_rust::TorchError::TensorError("Length mismatch".to_string()));
    }
    
    let asymmetric = y_true.iter()
        .zip(y_pred.iter())
        .map(|(&t, &p)| {
            let diff = t - p;
            if diff >= 0.0 {
                alpha * diff.powi(2) // Underestimation penalty
            } else {
                beta * diff.powi(2) // Overestimation penalty
            }
        })
        .sum::<f32>() / y_true.len() as f32;
    
    Ok(asymmetric)
}

fn quantile_loss(y_true: &[f32], y_pred: &[f32], tau: f32) -> Result<f32> {
    if y_true.len() != y_pred.len() {
        return Err(libtorch_rust::TorchError::TensorError("Length mismatch".to_string()));
    }
    
    let quantile = y_true.iter()
        .zip(y_pred.iter())
        .map(|(&t, &p)| {
            let diff = t - p;
            if diff >= 0.0 {
                tau * diff
            } else {
                (tau - 1.0) * diff
            }
        })
        .sum::<f32>() / y_true.len() as f32;
    
    Ok(quantile)
}

// Multi-class Loss Functions
fn categorical_cross_entropy(y_true: &[usize], y_pred_logits: &[Vec<f32>], n_classes: usize) -> Result<f32> {
    if y_true.len() != y_pred_logits.len() {
        return Err(libtorch_rust::TorchError::TensorError("Length mismatch".to_string()));
    }
    
    let mut total_loss = 0.0;
    
    for (&true_class, pred_logits) in y_true.iter().zip(y_pred_logits.iter()) {
        // Convert logits to probabilities using softmax
        let max_logit = pred_logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let exp_logits: Vec<f32> = pred_logits.iter().map(|&x| (x - max_logit).exp()).collect();
        let sum_exp = exp_logits.iter().sum::<f32>();
        let probabilities: Vec<f32> = exp_logits.iter().map(|&x| x / sum_exp).collect();
        
        // Cross-entropy loss
        let prob_true_class = probabilities[true_class].max(1e-7); // Avoid log(0)
        total_loss -= prob_true_class.ln();
    }
    
    Ok(total_loss / y_true.len() as f32)
}

fn sparse_categorical_cross_entropy(y_true: &[usize], y_pred_logits: &[Vec<f32>], _n_classes: usize) -> Result<f32> {
    // Same as categorical cross-entropy for sparse labels
    categorical_cross_entropy(y_true, y_pred_logits, _n_classes)
}

// Utility functions
fn add_outliers(y_true: &[f32], y_pred: &[f32]) -> Result<(Vec<f32>, Vec<f32>)> {
    let mut y_true_outliers = y_true.to_vec();
    let mut y_pred_outliers = y_pred.to_vec();
    
    // Add a few outliers
    let outlier_indices = [10, 25, 50, 75];
    for &idx in &outlier_indices {
        if idx < y_pred_outliers.len() {
            y_pred_outliers[idx] += 10.0; // Large error
        }
    }
    
    Ok((y_true_outliers, y_pred_outliers))
}

fn analyze_loss_behavior() -> Result<()> {
    println!("  Loss behavior for different error magnitudes:");
    println!("  Error |   MSE   |   MAE   |  Huber  | Log-Cosh");
    println!("  ------|---------|---------|---------|----------");
    
    let errors = [0.1, 0.5, 1.0, 2.0, 5.0];
    
    for &err in &errors {
        let y_true = vec![0.0];
        let y_pred = vec![err];
        
        let mse = mean_squared_error(&y_true, &y_pred)?;
        let mae = mean_absolute_error(&y_true, &y_pred)?;
        let huber = huber_loss(&y_true, &y_pred, 1.0)?;
        let log_cosh = log_cosh_loss(&y_true, &y_pred)?;
        
        println!("  {:5.1} | {:7.3} | {:7.3} | {:7.3} | {:8.3}", 
                err, mse, mae, huber, log_cosh);
    }
    
    Ok(())
}

fn analyze_gradients(y_true: &[f32], y_pred: &[f32]) -> Result<()> {
    println!("  Gradient analysis (first 5 samples):");
    println!("  Sample |  Error  | MSE Grad | MAE Grad | Huber Grad");
    println!("  -------|---------|----------|----------|----------");
    
    for i in 0..5.min(y_true.len()) {
        let error = y_true[i] - y_pred[i];
        
        // Gradients w.r.t. prediction
        let mse_grad = -2.0 * error; // d/dy_pred MSE = -2(y_true - y_pred)
        let mae_grad = if error > 0.0 { -1.0 } else { 1.0 }; // d/dy_pred MAE = sign(y_pred - y_true)
        let huber_grad = if error.abs() <= 1.0 { -error } else { -error.signum() };
        
        println!("  {:6} | {:7.3} | {:8.3} | {:8.3} | {:10.3}", 
                i + 1, error, mse_grad, mae_grad, huber_grad);
    }
    
    Ok(())
}

fn simulate_training_with_different_losses() -> Result<()> {
    println!("  Simulating training convergence with different losses:");
    
    let initial_error = 5.0;
    let epochs = 10;
    let learning_rate = 0.1;
    
    println!("  Epoch |   MSE   |   MAE   |  Huber");
    println!("  ------|---------|---------|--------");
    
    for epoch in 0..epochs {
        // Simulate prediction getting closer to target
        let current_error = initial_error * (0.8_f32).powi(epoch);
        
        let y_true = vec![0.0];
        let y_pred = vec![current_error];
        
        let mse = mean_squared_error(&y_true, &y_pred)?;
        let mae = mean_absolute_error(&y_true, &y_pred)?;
        let huber = huber_loss(&y_true, &y_pred, 1.0)?;
        
        println!("  {:5} | {:7.4} | {:7.4} | {:6.4}", 
                epoch, mse, mae, huber);
    }
    
    Ok(())
}

fn print_loss_function_guide() {
    println!("  Regression tasks:");
    println!("    - MSE: Standard choice, sensitive to outliers");
    println!("    - MAE: Robust to outliers, but non-differentiable at 0");
    println!("    - Huber: Balanced between MSE and MAE");
    println!("    - Log-Cosh: Smooth approximation to MAE");
    
    println!("  Classification tasks:");
    println!("    - Cross-Entropy: Standard for probabilistic outputs");
    println!("    - Hinge: Good for SVMs and margin-based learning");
    println!("    - Focal: Handles class imbalance");
    
    println!("  Custom scenarios:");
    println!("    - Quantile: For uncertainty estimation");
    println!("    - Asymmetric: When over/under-estimation have different costs");
    println!("    - Weighted: For handling sample importance");
}