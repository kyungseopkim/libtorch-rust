use libtorch_rust::{Tensor, Result};
use std::collections::HashMap;

fn main() -> Result<()> {
    println!("📚 LibTorch Rust - Dataset Example");
    println!("Demonstrating data loading and batching");
    println!("=====================================");
    
    // 1. Simple Dataset Creation
    println!("\n1. Creating a simple dataset:");
    let dataset = create_synthetic_dataset(1000)?;
    println!("  Dataset size: {} samples", dataset.features.len());
    println!("  Feature dimensions: {:?}", dataset.features[0].shape());
    println!("  Label shape: {:?}", dataset.labels[0].shape());
    
    // Display some samples
    println!("\n  First 5 samples:");
    for i in 0..5 {
        let feature_data = get_tensor_data(&dataset.features[i], 2)?;
        let label_data = get_tensor_data(&dataset.labels[i], 1)?;
        println!("    Sample {}: features = [{:.3}, {:.3}], label = {:.0}", 
                i + 1, feature_data[0], feature_data[1], label_data[0]);
    }
    
    // 2. Dataset Splitting
    println!("\n2. Splitting dataset:");
    let (train_dataset, test_dataset) = split_dataset(dataset, 0.8)?;
    println!("  Training samples: {}", train_dataset.features.len());
    println!("  Test samples: {}", test_dataset.features.len());
    
    // 3. Data Batching
    println!("\n3. Creating data batches:");
    let batch_size = 32;
    let train_batches = create_batches(&train_dataset, batch_size)?;
    println!("  Batch size: {}", batch_size);
    println!("  Number of batches: {}", train_batches.len());
    
    // Display batch information
    for (i, batch) in train_batches.iter().take(3).enumerate() {
        println!("  Batch {}: features shape = {:?}, labels shape = {:?}", 
                i + 1, batch.features.shape(), batch.labels.shape());
    }
    
    // 4. Data Shuffling
    println!("\n4. Data shuffling demonstration:");
    let mut shuffled_dataset = train_dataset.clone();
    shuffle_dataset(&mut shuffled_dataset)?;
    println!("  Dataset shuffled successfully");
    
    // Show that order changed
    println!("  First 3 samples after shuffle:");
    for i in 0..3 {
        let feature_data = get_tensor_data(&shuffled_dataset.features[i], 2)?;
        let label_data = get_tensor_data(&shuffled_dataset.labels[i], 1)?;
        println!("    Sample {}: features = [{:.3}, {:.3}], label = {:.0}", 
                i + 1, feature_data[0], feature_data[1], label_data[0]);
    }
    
    // 5. Data Normalization
    println!("\n5. Data normalization:");
    let (_normalized_features, stats) = normalize_features(&train_dataset.features)?;
    println!("  Feature statistics:");
    println!("    Mean: [{:.3}, {:.3}]", stats.mean[0], stats.mean[1]);
    println!("    Std:  [{:.3}, {:.3}]", stats.std[0], stats.std[1]);
    
    // 6. Different Dataset Types
    println!("\n6. Different dataset types:");
    
    // Image-like dataset (3D tensors)
    let image_dataset = create_image_dataset(100, 28, 28, 10)?;
    println!("  Image dataset:");
    println!("    Samples: {}", image_dataset.features.len());
    println!("    Image shape: {:?}", image_dataset.features[0].shape());
    println!("    Classes: {}", image_dataset.num_classes);
    
    // Time series dataset
    let time_series_dataset = create_time_series_dataset(200, 50)?;
    println!("  Time series dataset:");
    println!("    Samples: {}", time_series_dataset.sequences.len());
    println!("    Sequence length: {:?}", time_series_dataset.sequences[0].shape());
    
    // 7. Data Augmentation Simulation
    println!("\n7. Data augmentation:");
    let augmented_batch = augment_batch(&train_batches[0])?;
    println!("  Original batch size: {}", train_batches[0].features.shape()[0]);
    println!("  Augmented batch size: {}", augmented_batch.features.shape()[0]);
    
    // 8. Custom Data Loader
    println!("\n8. Custom data loader:");
    let data_loader = DataLoader::new(train_dataset.clone(), 16, true)?;
    println!("  Data loader created with batch size 16");
    println!("  Shuffle enabled: true");
    
    // Simulate iterating through epochs
    for epoch in 1..=3 {
        println!("  Epoch {}: {} batches available", epoch, data_loader.num_batches());
        
        // Show first batch info for each epoch
        let first_batch = data_loader.get_batch(0)?;
        let sample_features = get_tensor_data(&first_batch.features, 2)?;
        println!("    First batch, first sample: [{:.3}, {:.3}]", 
                sample_features[0], sample_features[1]);
    }
    
    // 9. Memory Usage Information
    println!("\n9. Memory usage information:");
    calculate_memory_usage(&train_dataset)?;
    
    // 10. Dataset Statistics
    println!("\n10. Dataset statistics:");
    analyze_dataset_distribution(&train_dataset)?;
    
    println!("\n✅ Dataset example completed!");
    println!("💡 This demonstrates various data handling patterns:");
    println!("   - Dataset creation and splitting");
    println!("   - Batching and shuffling");
    println!("   - Data normalization");
    println!("   - Different data types (tabular, image, time-series)");
    println!("   - Data augmentation concepts");
    
    Ok(())
}

// Dataset structures
#[derive(Clone)]
struct Dataset {
    features: Vec<Tensor>,
    labels: Vec<Tensor>,
}

struct Batch {
    features: Tensor,
    labels: Tensor,
}

struct ImageDataset {
    features: Vec<Tensor>,
    #[allow(dead_code)]
    labels: Vec<Tensor>,
    num_classes: usize,
}

struct TimeSeriesDataset {
    sequences: Vec<Tensor>,
    #[allow(dead_code)]
    targets: Vec<Tensor>,
}

struct FeatureStats {
    mean: Vec<f32>,
    std: Vec<f32>,
}

struct DataLoader {
    dataset: Dataset,
    batch_size: usize,
    #[allow(dead_code)]
    shuffle: bool,
    indices: Vec<usize>,
}

impl DataLoader {
    fn new(dataset: Dataset, batch_size: usize, shuffle: bool) -> Result<Self> {
        let mut indices: Vec<usize> = (0..dataset.features.len()).collect();
        if shuffle {
            // Simple shuffle using pseudo-random
            for i in 0..indices.len() {
                let j = (i * 17 + 31) % indices.len();
                indices.swap(i, j);
            }
        }
        
        Ok(DataLoader {
            dataset,
            batch_size,
            shuffle,
            indices,
        })
    }
    
    fn num_batches(&self) -> usize {
        (self.dataset.features.len() + self.batch_size - 1) / self.batch_size
    }
    
    fn get_batch(&self, batch_idx: usize) -> Result<Batch> {
        let start_idx = batch_idx * self.batch_size;
        let end_idx = (start_idx + self.batch_size).min(self.dataset.features.len());
        
        if start_idx >= self.dataset.features.len() {
            return Err(libtorch_rust::TorchError::TensorError("Batch index out of range".to_string()));
        }
        
        // Get batch indices
        let batch_indices = &self.indices[start_idx..end_idx];
        
        // Create batch tensors
        let mut batch_features = Vec::new();
        let mut batch_labels = Vec::new();
        
        for &idx in batch_indices {
            let feature_data = get_tensor_data(&self.dataset.features[idx], 2)?;
            let label_data = get_tensor_data(&self.dataset.labels[idx], 1)?;
            
            batch_features.extend(feature_data);
            batch_labels.extend(label_data);
        }
        
        let features = Tensor::new(batch_features, vec![batch_indices.len() as i64, 2])?;
        let labels = Tensor::new(batch_labels, vec![batch_indices.len() as i64, 1])?;
        
        Ok(Batch { features, labels })
    }
}

// Dataset creation functions
fn create_synthetic_dataset(num_samples: usize) -> Result<Dataset> {
    let mut features = Vec::new();
    let mut labels = Vec::new();
    
    for i in 0..num_samples {
        // Create 2D features with some pattern
        let x1 = (i as f32 * 0.01) % 10.0 - 5.0; // Range: [-5, 5]
        let x2 = ((i as f32 * 0.017) % 8.0) - 4.0; // Range: [-4, 4]
        
        // Simple classification: label = 1 if x1 + x2 > 0, else 0
        let label = if x1 + x2 > 0.0 { 1.0 } else { 0.0 };
        
        let feature_tensor = Tensor::new(vec![x1, x2], vec![2])?;
        let label_tensor = Tensor::new(vec![label], vec![1])?;
        
        features.push(feature_tensor);
        labels.push(label_tensor);
    }
    
    Ok(Dataset { features, labels })
}

fn create_image_dataset(num_samples: usize, height: i64, width: i64, num_classes: usize) -> Result<ImageDataset> {
    let mut features = Vec::new();
    let mut labels = Vec::new();
    
    for i in 0..num_samples {
        // Create synthetic image data
        let pixel_count = (height * width) as usize;
        let image_data: Vec<f32> = (0..pixel_count)
            .map(|j| ((i + j) as f32 * 0.001) % 1.0) // Normalized pixel values
            .collect();
        
        let image_tensor = Tensor::new(image_data, vec![1, height, width])?; // 1 channel
        let label = (i % num_classes) as f32;
        let label_tensor = Tensor::new(vec![label], vec![1])?;
        
        features.push(image_tensor);
        labels.push(label_tensor);
    }
    
    Ok(ImageDataset { features, labels, num_classes })
}

fn create_time_series_dataset(num_samples: usize, sequence_length: usize) -> Result<TimeSeriesDataset> {
    let mut sequences = Vec::new();
    let mut targets = Vec::new();
    
    for i in 0..num_samples {
        // Create synthetic time series (sine wave with noise)
        let sequence_data: Vec<f32> = (0..sequence_length)
            .map(|t| {
                let time = (i + t) as f32 * 0.1;
                time.sin() + ((i + t) as f32 * 0.123) % 0.2 - 0.1 // Add noise
            })
            .collect();
        
        // Target is the next value in the sequence
        let target = (i + sequence_length) as f32 * 0.1;
        let target_value = target.sin();
        
        let sequence_tensor = Tensor::new(sequence_data, vec![sequence_length as i64])?;
        let target_tensor = Tensor::new(vec![target_value], vec![1])?;
        
        sequences.push(sequence_tensor);
        targets.push(target_tensor);
    }
    
    Ok(TimeSeriesDataset { sequences, targets })
}

// Dataset manipulation functions
fn split_dataset(dataset: Dataset, train_ratio: f32) -> Result<(Dataset, Dataset)> {
    let total_samples = dataset.features.len();
    let train_size = (total_samples as f32 * train_ratio) as usize;
    
    let train_features = dataset.features[..train_size].to_vec();
    let train_labels = dataset.labels[..train_size].to_vec();
    let test_features = dataset.features[train_size..].to_vec();
    let test_labels = dataset.labels[train_size..].to_vec();
    
    let train_dataset = Dataset {
        features: train_features,
        labels: train_labels,
    };
    
    let test_dataset = Dataset {
        features: test_features,
        labels: test_labels,
    };
    
    Ok((train_dataset, test_dataset))
}

fn create_batches(dataset: &Dataset, batch_size: usize) -> Result<Vec<Batch>> {
    let mut batches = Vec::new();
    let num_samples = dataset.features.len();
    
    for start_idx in (0..num_samples).step_by(batch_size) {
        let end_idx = (start_idx + batch_size).min(num_samples);
        let current_batch_size = end_idx - start_idx;
        
        let mut batch_features = Vec::new();
        let mut batch_labels = Vec::new();
        
        for i in start_idx..end_idx {
            let feature_data = get_tensor_data(&dataset.features[i], 2)?;
            let label_data = get_tensor_data(&dataset.labels[i], 1)?;
            
            batch_features.extend(feature_data);
            batch_labels.extend(label_data);
        }
        
        let features_tensor = Tensor::new(batch_features, vec![current_batch_size as i64, 2])?;
        let labels_tensor = Tensor::new(batch_labels, vec![current_batch_size as i64, 1])?;
        
        batches.push(Batch {
            features: features_tensor,
            labels: labels_tensor,
        });
    }
    
    Ok(batches)
}

fn shuffle_dataset(dataset: &mut Dataset) -> Result<()> {
    let n = dataset.features.len();
    
    // Simple pseudo-random shuffle
    for i in 0..n {
        let j = (i * 17 + 31) % n;
        dataset.features.swap(i, j);
        dataset.labels.swap(i, j);
    }
    
    Ok(())
}

fn normalize_features(features: &[Tensor]) -> Result<(Vec<Tensor>, FeatureStats)> {
    let num_features = 2; // Assuming 2D features
    let mut means = vec![0.0; num_features];
    let mut stds = vec![0.0; num_features];
    
    // Calculate means
    for feature_tensor in features {
        let data = get_tensor_data(feature_tensor, num_features)?;
        for (i, &value) in data.iter().enumerate() {
            means[i] += value;
        }
    }
    
    for mean in &mut means {
        *mean /= features.len() as f32;
    }
    
    // Calculate standard deviations
    for feature_tensor in features {
        let data = get_tensor_data(feature_tensor, num_features)?;
        for (i, &value) in data.iter().enumerate() {
            stds[i] += (value - means[i]).powi(2);
        }
    }
    
    for std in &mut stds {
        *std = (*std / features.len() as f32).sqrt();
        if *std == 0.0 {
            *std = 1.0; // Avoid division by zero
        }
    }
    
    // Normalize features
    let mut normalized_features = Vec::new();
    for feature_tensor in features {
        let data = get_tensor_data(feature_tensor, num_features)?;
        let normalized_data: Vec<f32> = data.iter().enumerate()
            .map(|(i, &value)| (value - means[i]) / stds[i])
            .collect();
        
        normalized_features.push(Tensor::new(normalized_data, vec![num_features as i64])?);
    }
    
    let stats = FeatureStats {
        mean: means,
        std: stds,
    };
    
    Ok((normalized_features, stats))
}

fn augment_batch(batch: &Batch) -> Result<Batch> {
    // Simple augmentation: add noise to features
    let batch_size = batch.features.shape()[0] as usize;
    let feature_dim = batch.features.shape()[1] as usize;
    
    let original_features = get_tensor_data(&batch.features, batch_size * feature_dim)?;
    let original_labels = get_tensor_data(&batch.labels, batch_size)?;
    
    let mut augmented_features = original_features.clone();
    let mut augmented_labels = original_labels.clone();
    
    // Add noisy versions
    for i in 0..batch_size {
        for j in 0..feature_dim {
            let noise = ((i + j) as f32 * 0.1) % 0.2 - 0.1; // Simple noise
            let original_value = original_features[i * feature_dim + j];
            augmented_features.push(original_value + noise);
        }
        augmented_labels.push(original_labels[i]);
    }
    
    let augmented_batch_size = batch_size * 2;
    let features = Tensor::new(augmented_features, vec![augmented_batch_size as i64, feature_dim as i64])?;
    let labels = Tensor::new(augmented_labels, vec![augmented_batch_size as i64, 1])?;
    
    Ok(Batch { features, labels })
}

// Utility functions
fn get_tensor_data(tensor: &Tensor, expected_size: usize) -> Result<Vec<f32>> {
    let data_ptr = tensor.data_ptr();
    let mut data = Vec::new();
    
    for i in 0..expected_size {
        let value = unsafe { *data_ptr.add(i) };
        data.push(value);
    }
    
    Ok(data)
}

fn calculate_memory_usage(dataset: &Dataset) -> Result<()> {
    let num_samples = dataset.features.len();
    let feature_size = 2 * 4; // 2 floats * 4 bytes each
    let label_size = 1 * 4; // 1 float * 4 bytes
    
    let total_memory = num_samples * (feature_size + label_size);
    
    println!("  Dataset memory usage:");
    println!("    Samples: {}", num_samples);
    println!("    Memory per sample: {} bytes", feature_size + label_size);
    println!("    Total memory: {:.2} KB", total_memory as f32 / 1024.0);
    
    Ok(())
}

fn analyze_dataset_distribution(dataset: &Dataset) -> Result<()> {
    let mut class_counts = HashMap::new();
    
    for label_tensor in &dataset.labels {
        let label_data = get_tensor_data(label_tensor, 1)?;
        let label = label_data[0] as i32;
        *class_counts.entry(label).or_insert(0) += 1;
    }
    
    println!("  Class distribution:");
    for (class, count) in class_counts {
        let percentage = (count as f32 / dataset.labels.len() as f32) * 100.0;
        println!("    Class {}: {} samples ({:.1}%)", class, count, percentage);
    }
    
    Ok(())
}