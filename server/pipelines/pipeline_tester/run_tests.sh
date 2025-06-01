#!/bin/bash

# Pipeline Tester Runner Script
# This script provides easy commands to run various pipeline tests

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if we're in the right directory
check_directory() {
    if [[ ! -f "test_controlnet_pipeline.py" ]]; then
        print_error "Please run this script from the pipeline_tester directory"
        exit 1
    fi
}

# Function to check Python environment
check_python() {
    if ! command -v python3 &> /dev/null; then
        print_error "Python3 is not installed or not in PATH"
        exit 1
    fi
    
    print_status "Using Python: $(which python3)"
    print_status "Python version: $(python3 --version)"
}

# Function to show usage
show_usage() {
    echo "Pipeline Tester Runner"
    echo "====================="
    echo ""
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  quick          Run a quick test (3 iterations, no output saving)"
    echo "  standard       Run standard test (5 iterations, save outputs)"
    echo "  extensive      Run extensive test (10 iterations, save outputs)"
    echo "  custom         Run custom test scenarios"
    echo "  benchmark      Run benchmark test (20 iterations for performance)"
    echo "  help           Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 quick"
    echo "  $0 standard"
    echo "  $0 extensive"
    echo "  $0 custom"
    echo "  $0 benchmark"
    echo ""
}

# Function to run quick test
run_quick_test() {
    print_status "Running quick test (3 iterations, no output saving)..."
    python3 test_controlnet_pipeline.py --num-iterations 3
}

# Function to run standard test
run_standard_test() {
    print_status "Running standard test (5 iterations, save outputs)..."
    python3 test_controlnet_pipeline.py --num-iterations 5 --save-outputs
}

# Function to run extensive test
run_extensive_test() {
    print_status "Running extensive test (10 iterations, save outputs)..."
    python3 test_controlnet_pipeline.py --num-iterations 10 --save-outputs --output-dir extensive_test_outputs
}

# Function to run custom test
run_custom_test() {
    print_status "Running custom test scenarios..."
    if [[ -f "example_custom_test.py" ]]; then
        python3 example_custom_test.py
    else
        print_error "example_custom_test.py not found"
        exit 1
    fi
}

# Function to run benchmark test
run_benchmark_test() {
    print_status "Running benchmark test (20 iterations for performance measurement)..."
    python3 test_controlnet_pipeline.py --num-iterations 20 --output-dir benchmark_outputs
}

# Function to clean up old test outputs
cleanup_outputs() {
    print_status "Cleaning up old test outputs..."
    
    if [[ -d "test_outputs" ]]; then
        rm -rf test_outputs
        print_status "Removed test_outputs directory"
    fi
    
    if [[ -d "extensive_test_outputs" ]]; then
        rm -rf extensive_test_outputs
        print_status "Removed extensive_test_outputs directory"
    fi
    
    if [[ -d "benchmark_outputs" ]]; then
        rm -rf benchmark_outputs
        print_status "Removed benchmark_outputs directory"
    fi
    
    # Remove individual test files
    rm -f custom_test_*.png strength_test_*.png control_image.png
    
    print_success "Cleanup completed"
}

# Main script logic
main() {
    check_directory
    check_python
    
    case "${1:-help}" in
        "quick")
            run_quick_test
            ;;
        "standard")
            run_standard_test
            ;;
        "extensive")
            run_extensive_test
            ;;
        "custom")
            run_custom_test
            ;;
        "benchmark")
            run_benchmark_test
            ;;
        "cleanup")
            cleanup_outputs
            ;;
        "help"|*)
            show_usage
            ;;
    esac
}

# Trap to handle interruption
trap 'print_warning "Test interrupted by user"; exit 1' INT

# Run main function
main "$@" 