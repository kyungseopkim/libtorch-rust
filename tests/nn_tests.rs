use libtorch_rust::{Tensor, nn::{Linear, Module}, Result};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_layer_creation() -> Result<()> {
        let linear = Linear::new(4, 8)?;
        
        assert_eq!(linear.in_features(), 4);
        assert_eq!(linear.out_features(), 8);
        
        Ok(())
    }
    
    #[test]
    fn test_linear_forward() -> Result<()> {
        let linear = Linear::new(3, 2)?;
        let input = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3])?;
        
        let output = linear.forward(&input)?;
        assert_eq!(output.shape(), &[1, 2]);
        
        Ok(())
    }
    
    #[test]
    fn test_sequential_layers() -> Result<()> {
        let input = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4])?;
        
        let linear1 = Linear::new(4, 8)?;
        let hidden = linear1.forward(&input)?;
        assert_eq!(hidden.shape(), &[1, 8]);
        
        let relu_output = hidden.clamp_min(0.0)?;
        assert_eq!(relu_output.shape(), &[1, 8]);
        
        let linear2 = Linear::new(8, 2)?;
        let output = linear2.forward(&relu_output)?;
        assert_eq!(output.shape(), &[1, 2]);
        
        Ok(())
    }
    
    #[test]
    fn test_batch_processing() -> Result<()> {
        let linear = Linear::new(3, 2)?;
        let batch_input = Tensor::new(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 
            vec![2, 3]
        )?;
        
        let output = linear.forward(&batch_input)?;
        assert_eq!(output.shape(), &[2, 2]);
        
        Ok(())
    }
}