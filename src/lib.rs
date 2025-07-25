pub mod error;
pub mod tensor;
pub mod simple_test;
pub mod nn;

pub use tensor::Tensor;
pub use error::{TorchError, Result};

// include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
