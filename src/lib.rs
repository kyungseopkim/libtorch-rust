pub mod error;
pub mod tensor;

pub use tensor::Tensor;
pub use error::{TorchError, Result};

// include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
