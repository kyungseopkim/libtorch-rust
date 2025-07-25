use libtorch_rust::{Tensor, Result};

fn main() -> Result<()> {
    println!("📊 LibTorch Rust - Time Series Prediction");
    println!("Predicting future values in a time series");
    println!("========================================");
    
    // Generate synthetic time series data: y = sin(t) + 0.5*sin(3t) + noise
    println!("\n1. Generating synthetic time series data...");
    let num_points = 200;
    let mut time_series = Vec::new();
    let mut time_points = Vec::new();
    
    for i in 0..num_points {
        let t = (i as f32) * 0.1; // Time step = 0.1
        let noise = ((i as f32 * 0.123) % 1.0 - 0.5) * 0.1; // Simple pseudo-noise
        let y = (t).sin() + 0.5 * (3.0 * t).sin() + noise;
        
        time_series.push(y);
        time_points.push(t);
    }
    
    let _series_tensor = Tensor::new(time_series.clone(), vec![num_points as i64])?;
    println!("  Generated {} time series points", num_points);
    println!("  Data range: [{:.3}, {:.3}]", 
             time_series.iter().fold(f32::INFINITY, |a, &b| a.min(b)),
             time_series.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)));
    
    // Display first few points
    println!("  First 10 points:");
    for i in 0..10 {
        println!("    t = {:5.2}, y = {:7.4}", time_points[i], time_series[i]);
    }
    
    // Create sliding window dataset
    println!("\n2. Creating sliding window dataset...");
    let window_size = 10;
    let prediction_horizon = 1;
    
    let (x_data, y_data) = create_sliding_windows(&time_series, window_size, prediction_horizon)?;
    println!("  Window size: {}", window_size);
    println!("  Prediction horizon: {}", prediction_horizon);
    println!("  Number of training samples: {}", x_data.len() / window_size);
    
    // Split data into train and test
    let split_ratio = 0.8;
    let split_point = (x_data.len() as f32 * split_ratio) as usize;
    let split_point = (split_point / window_size) * window_size; // Align with window boundaries
    
    let x_train_data = x_data[..split_point].to_vec();
    let _y_train_data = y_data[..split_point/window_size].to_vec();
    let x_test_data = x_data[split_point..].to_vec();
    let y_test_data = y_data[split_point/window_size..].to_vec();
    
    println!("  Training samples: {}", x_train_data.len() / window_size);
    println!("  Test samples: {}", x_test_data.len() / window_size);
    
    // Simple prediction model using moving average
    println!("\n3. Implementing simple prediction models...");
    
    // Model 1: Moving Average
    let ma_predictions = predict_moving_average(&x_test_data, window_size)?;
    let ma_mse = calculate_mse(&ma_predictions, &y_test_data)?;
    println!("  Moving Average Model:");
    println!("    MSE: {:.6}", ma_mse);
    
    // Model 2: Linear Trend
    let trend_predictions = predict_linear_trend(&x_test_data, window_size)?;
    let trend_mse = calculate_mse(&trend_predictions, &y_test_data)?;
    println!("  Linear Trend Model:");
    println!("    MSE: {:.6}", trend_mse);
    
    // Model 3: Weighted Moving Average
    let wma_predictions = predict_weighted_moving_average(&x_test_data, window_size)?;
    let wma_mse = calculate_mse(&wma_predictions, &y_test_data)?;
    println!("  Weighted Moving Average Model:");
    println!("    MSE: {:.6}", wma_mse);
    
    // Show prediction examples
    println!("\n4. Prediction examples:");
    println!("  Sample |  Actual  | MovAvg | Trend  | WtdAvg | Best");
    println!("  -------|----------|--------|--------|--------|------");
    
    let num_examples = 10.min(y_test_data.len());
    for i in 0..num_examples {
        let actual = y_test_data[i];
        let ma_pred = ma_predictions[i];
        let trend_pred = trend_predictions[i];
        let wma_pred = wma_predictions[i];
        
        // Find best prediction
        let ma_err = (actual - ma_pred).abs();
        let trend_err = (actual - trend_pred).abs();
        let wma_err = (actual - wma_pred).abs();
        
        let best = if ma_err <= trend_err && ma_err <= wma_err {
            "MA"
        } else if trend_err <= wma_err {
            "Trend"
        } else {
            "WMA"
        };
        
        println!("  {:6} | {:8.4} | {:6.3} | {:6.3} | {:6.3} | {}", 
                i + 1, actual, ma_pred, trend_pred, wma_pred, best);
    }
    
    // Analyze prediction errors
    println!("\n5. Error analysis:");
    analyze_prediction_errors(&ma_predictions, &trend_predictions, &wma_predictions, &y_test_data)?;
    
    // Demonstrate multi-step prediction
    println!("\n6. Multi-step ahead prediction:");
    demonstrate_multistep_prediction(&time_series, window_size)?;
    
    // Pattern detection
    println!("\n7. Pattern detection:");
    detect_patterns(&time_series)?;
    
    println!("\n✅ Time series prediction example completed!");
    println!("💡 This demonstrates various approaches to predicting future");
    println!("   values in sequential data using different algorithms");
    
    Ok(())
}

fn create_sliding_windows(data: &[f32], window_size: usize, horizon: usize) -> Result<(Vec<f32>, Vec<f32>)> {
    let mut x_data = Vec::new();
    let mut y_data = Vec::new();
    
    for i in 0..(data.len() - window_size - horizon + 1) {
        // Input window
        for j in 0..window_size {
            x_data.push(data[i + j]);
        }
        
        // Target (next value after the window)
        y_data.push(data[i + window_size + horizon - 1]);
    }
    
    Ok((x_data, y_data))
}

fn predict_moving_average(x_data: &[f32], window_size: usize) -> Result<Vec<f32>> {
    let mut predictions = Vec::new();
    let num_samples = x_data.len() / window_size;
    
    for i in 0..num_samples {
        let start_idx = i * window_size;
        let end_idx = start_idx + window_size;
        let window = &x_data[start_idx..end_idx];
        
        let avg = window.iter().sum::<f32>() / window_size as f32;
        predictions.push(avg);
    }
    
    Ok(predictions)
}

fn predict_linear_trend(x_data: &[f32], window_size: usize) -> Result<Vec<f32>> {
    let mut predictions = Vec::new();
    let num_samples = x_data.len() / window_size;
    
    for i in 0..num_samples {
        let start_idx = i * window_size;
        let end_idx = start_idx + window_size;
        let window = &x_data[start_idx..end_idx];
        
        // Simple linear trend: predict based on last two points
        if window_size >= 2 {
            let trend = window[window_size - 1] - window[window_size - 2];
            let prediction = window[window_size - 1] + trend;
            predictions.push(prediction);
        } else {
            predictions.push(window[0]);
        }
    }
    
    Ok(predictions)
}

fn predict_weighted_moving_average(x_data: &[f32], window_size: usize) -> Result<Vec<f32>> {
    let mut predictions = Vec::new();
    let num_samples = x_data.len() / window_size;
    
    for i in 0..num_samples {
        let start_idx = i * window_size;
        let end_idx = start_idx + window_size;
        let window = &x_data[start_idx..end_idx];
        
        // Weighted average: more weight to recent values
        let mut weighted_sum = 0.0;
        let mut weight_sum = 0.0;
        
        for (j, &value) in window.iter().enumerate() {
            let weight = (j + 1) as f32; // Linear increasing weights
            weighted_sum += weight * value;
            weight_sum += weight;
        }
        
        let prediction = weighted_sum / weight_sum;
        predictions.push(prediction);
    }
    
    Ok(predictions)
}

fn calculate_mse(predictions: &[f32], actuals: &[f32]) -> Result<f32> {
    if predictions.len() != actuals.len() {
        return Err(libtorch_rust::TorchError::TensorError("Length mismatch".to_string()));
    }
    
    let mse = predictions.iter()
        .zip(actuals.iter())
        .map(|(&pred, &actual)| (pred - actual).powi(2))
        .sum::<f32>() / predictions.len() as f32;
    
    Ok(mse)
}

fn analyze_prediction_errors(ma_pred: &[f32], trend_pred: &[f32], wma_pred: &[f32], actual: &[f32]) -> Result<()> {
    let ma_errors: Vec<f32> = ma_pred.iter().zip(actual.iter()).map(|(&p, &a)| (p - a).abs()).collect();
    let trend_errors: Vec<f32> = trend_pred.iter().zip(actual.iter()).map(|(&p, &a)| (p - a).abs()).collect();
    let wma_errors: Vec<f32> = wma_pred.iter().zip(actual.iter()).map(|(&p, &a)| (p - a).abs()).collect();
    
    println!("  Mean Absolute Error:");
    println!("    Moving Average: {:.6}", ma_errors.iter().sum::<f32>() / ma_errors.len() as f32);
    println!("    Linear Trend:   {:.6}", trend_errors.iter().sum::<f32>() / trend_errors.len() as f32);
    println!("    Weighted MA:    {:.6}", wma_errors.iter().sum::<f32>() / wma_errors.len() as f32);
    
    println!("  Max Absolute Error:");
    println!("    Moving Average: {:.6}", ma_errors.iter().fold(0.0f32, |a, &b| a.max(b)));
    println!("    Linear Trend:   {:.6}", trend_errors.iter().fold(0.0f32, |a, &b| a.max(b)));
    println!("    Weighted MA:    {:.6}", wma_errors.iter().fold(0.0f32, |a, &b| a.max(b)));
    
    Ok(())
}

fn demonstrate_multistep_prediction(data: &[f32], window_size: usize) -> Result<()> {
    let start_idx = data.len() - window_size - 5; // Use last part of data
    let seed_window = &data[start_idx..start_idx + window_size];
    
    println!("  Predicting 5 steps ahead using seed window:");
    println!("  Step | Prediction | Method");
    println!("  -----|------------|--------");
    
    let mut current_window = seed_window.to_vec();
    
    for step in 1..=5 {
        // Simple prediction: weighted average of last 3 values
        let last_3 = &current_window[current_window.len() - 3..];
        let weights = [0.5, 0.3, 0.2];
        let prediction = last_3.iter().zip(weights.iter()).map(|(&val, &w)| val * w).sum::<f32>();
        
        println!("  {:4} | {:10.4} | Weighted MA", step, prediction);
        
        // Add prediction to window for next step (remove oldest)
        current_window.remove(0);
        current_window.push(prediction);
    }
    
    Ok(())
}

fn detect_patterns(data: &[f32]) -> Result<()> {
    println!("  Analyzing patterns in the time series:");
    
    // Calculate basic statistics
    let mean = data.iter().sum::<f32>() / data.len() as f32;
    let variance = data.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / data.len() as f32;
    let std_dev = variance.sqrt();
    
    println!("    Mean: {:.3}", mean);
    println!("    Std Dev: {:.3}", std_dev);
    
    // Detect trend
    let first_half_mean = data[..data.len()/2].iter().sum::<f32>() / (data.len()/2) as f32;
    let second_half_mean = data[data.len()/2..].iter().sum::<f32>() / (data.len()/2) as f32;
    
    let trend_direction = if second_half_mean > first_half_mean + 0.1 {
        "Increasing"
    } else if second_half_mean < first_half_mean - 0.1 {
        "Decreasing"
    } else {
        "Stable"
    };
    
    println!("    Trend: {}", trend_direction);
    
    // Simple periodicity detection (basic autocorrelation)
    let potential_periods = [10, 20, 30, 50];
    println!("    Potential periodicities:");
    
    for &period in &potential_periods {
        if period < data.len() / 2 {
            let correlation = calculate_autocorrelation(data, period);
            println!("      Period {}: correlation = {:.3}", period, correlation);
        }
    }
    
    Ok(())
}

fn calculate_autocorrelation(data: &[f32], lag: usize) -> f32 {
    if lag >= data.len() {
        return 0.0;
    }
    
    let n = data.len() - lag;
    let mean = data.iter().sum::<f32>() / data.len() as f32;
    
    let mut numerator = 0.0;
    let mut denominator = 0.0;
    
    for i in 0..n {
        let x_i = data[i] - mean;
        let x_lag = data[i + lag] - mean;
        numerator += x_i * x_lag;
        denominator += x_i * x_i;
    }
    
    if denominator == 0.0 {
        0.0
    } else {
        numerator / denominator
    }
}