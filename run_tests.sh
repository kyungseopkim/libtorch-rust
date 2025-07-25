#!/bin/bash
echo "🧪 Running LibTorch Rust Tests"
echo "=============================="

export DYLD_LIBRARY_PATH=../libtorch/lib

echo "Building library..."
cargo build --lib --release

if [ $? -eq 0 ]; then
    echo "✅ Build successful"
    echo ""
    echo "Running unit tests..."
    cargo test --lib --release simple_test
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "🎉 All tests passed successfully!"
        echo ""
        echo "Running full test suite..."
        cargo run --example test_runner --release
    else
        echo "❌ Some tests failed"
        exit 1
    fi
else
    echo "❌ Build failed"
    exit 1
fi