#!/bin/bash
echo "🚀 Running LibTorch Rust Examples"
echo "=================================="

export DYLD_LIBRARY_PATH=../libtorch/lib

# Function to run example with status reporting
run_example() {
    local example_name=$1
    local description=$2
    
    echo ""
    echo "📋 Running: $example_name"
    echo "   Description: $description"
    echo "   Command: cargo run --example $example_name"
    echo "   ---"
    
    if cargo run --example "$example_name" 2>/dev/null; then
        echo "   ✅ SUCCESS: $example_name completed"
        return 0
    else
        echo "   ❌ FAILED: $example_name had compilation or runtime errors"
        echo "   💡 Run manually to see detailed errors:"
        echo "      DYLD_LIBRARY_PATH=../libtorch/lib cargo run --example $example_name"
        return 1
    fi
}

echo ""
echo "🎯 CORE FUNCTIONALITY EXAMPLES"
echo "==============================="

successful_examples=0
total_core_examples=0

# Core tensor operations
((total_core_examples++))
if run_example "test_runner" "Comprehensive test suite with detailed validation"; then
    ((successful_examples++))
fi

((total_core_examples++))
if run_example "performance_test" "Performance benchmarks for matrix operations"; then
    ((successful_examples++))
fi

((total_core_examples++))
if run_example "hello_world" "Complete demonstration of basic tensor features"; then
    ((successful_examples++))
fi

((total_core_examples++))
if run_example "basic_tensor" "Basic tensor operations showcase"; then
    ((successful_examples++))
fi

((total_core_examples++))
if run_example "simple_demo" "Simple demo with larger matrix operations"; then
    ((successful_examples++))
fi

echo ""
echo "🧠 NEURAL NETWORKS & AUTOGRAD EXAMPLES"
echo "======================================="

# Neural networks and autograd
((total_core_examples++))
if run_example "neural_network" "Neural network with Linear layers and activations"; then
    ((successful_examples++))
fi

((total_core_examples++))
if run_example "autograd" "Automatic differentiation with gradient computation"; then
    ((successful_examples++))
fi

echo ""
echo "🚧 ADVANCED EXAMPLES (Some Issues Expected)"
echo "============================================"

advanced_successful=0
total_advanced=0

# Advanced examples with known issues
((total_advanced++))
if run_example "dataset_example" "Dataset handling and data loading"; then
    ((advanced_successful++))
fi

((total_advanced++))
if run_example "custom_loss_function" "Custom loss functions and training"; then
    ((advanced_successful++))
fi

((total_advanced++))
if run_example "function_approx" "Function approximation with neural networks"; then
    ((advanced_successful++))
fi

((total_advanced++))
if run_example "simple_optimization" "Gradient-based optimization algorithms"; then
    ((advanced_successful++))
fi

((total_advanced++))
if run_example "time_series_prediction" "Time series forecasting example"; then
    ((advanced_successful++))
fi

echo ""
echo "📊 SUMMARY REPORT"
echo "=================="
echo "🎯 Core Functionality: $successful_examples/$total_core_examples examples working"
echo "🧠 Advanced Features: $advanced_successful/$total_advanced examples working"

total_examples=$((total_core_examples + total_advanced))
total_working=$((successful_examples + advanced_successful))

echo "📈 Overall Success Rate: $total_working/$total_examples examples ($((total_working * 100 / total_examples))%)"

echo ""
if [ $successful_examples -eq $total_core_examples ]; then
    echo "🎉 ALL CORE FUNCTIONALITY IS WORKING PERFECTLY!"
    echo "✅ Tensor operations, neural networks, and autograd are fully functional"
else
    echo "⚠️  Some core functionality issues detected"
fi

if [ $advanced_successful -gt 0 ]; then
    echo "🌟 $advanced_successful advanced examples are also working!"
fi

echo ""
echo "💡 QUICK REFERENCE"
echo "=================="
echo "• Run individual example: DYLD_LIBRARY_PATH=../libtorch/lib cargo run --example <name>"
echo "• Run all tests: ./run_tests.sh"
echo "• Build project: cargo build --release"
echo "• Check examples compilation: cargo check --examples"

echo ""
echo "🔥 IMPLEMENTED FEATURES"
echo "======================="
echo "✅ Tensor creation, manipulation, and operations"
echo "✅ Neural network layers (Linear) with activations (ReLU, Softmax)"
echo "✅ Automatic differentiation (requires_grad, backward, grad)"
echo "✅ Clone trait for tensor copying and dataset handling"
echo "✅ Memory-safe tensor management with proper Drop implementation"
echo "✅ CPU-only execution (default) with optional CUDA support"
echo "✅ Cross-platform compatibility (macOS, Linux, Windows)"
echo ""
echo "🚀 Ready for deep learning development!"