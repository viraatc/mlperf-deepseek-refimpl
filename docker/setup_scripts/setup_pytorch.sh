#!/bin/bash
# Setup script for MLPerf DeepSeek evaluation environment - PyTorch Backend
# This script sets up the PyTorch backend with virtual environment activated

set -e  # Exit on error

# Source common functions
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

# Parse command line arguments
FORCE_REBUILD=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --force-rebuild)
            FORCE_REBUILD=true
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --force-rebuild    Force rebuild of MLPerf LoadGen from source"
            echo "  --help            Show this help message"
            echo ""
            echo "PyTorch Backend Setup:"
            echo "- Creates and activates virtual environment for all operations"
            echo "- Installs accuracy evaluation dependencies"
            echo "- Sets up MLPerf LoadGen"
            echo "- Virtual environment remains active after setup"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "=== Setting up MLPerf DeepSeek evaluation environment - PyTorch Backend ==="
echo "=== Virtual environment will be active for all PyTorch operations ==="

# Check if uv is installed
check_uv_installed

# Set the virtual environment directory
VENV_DIR="/work/.venv_pytorch"

# Setup virtual environment
setup_virtual_environment "$VENV_DIR"

# Activate the virtual environment for setup and subsequent use
echo "Activating virtual environment for PyTorch backend..."
source "$VENV_DIR/bin/activate"

# Install build dependencies
install_build_dependencies

# Apply patch to fix prm800k setup.py
patch_prm800k_setup

# Install evaluation requirements
install_evaluation_requirements

# Install MLPerf LoadGen
install_mlperf_loadgen "$FORCE_REBUILD"

# PyTorch-specific setup
echo ""
echo "=== PyTorch Backend-Specific Setup ==="

# Verify PyTorch is available (should be from base image)
if python3 -c "import torch" 2>/dev/null; then
    TORCH_VERSION=$(python3 -c "import torch; print(torch.__version__)")
    echo "PyTorch is available: version $TORCH_VERSION"
else
    echo "Warning: PyTorch not found in the environment"
fi

# Verify ref_dsinfer package is available
if python3 -c "import ref_dsinfer" 2>/dev/null; then
    echo "DeepSeek-V3 ref_dsinfer package is available"
else
    echo "Warning: ref_dsinfer package not found. Check PYTHONPATH and /opt/ref_dsinfer"
fi

# Model download and conversion
echo ""
echo "=== Model Download and Conversion ==="
echo "Downloading DeepSeek-R1 model..."
HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download deepseek-ai/DeepSeek-R1 --local-dir /raid/data/viraatc/models/deepseek-ai_DeepSeek-R1

echo "Converting model to inference format..."
uv run python /opt/ref_dsinfer/inference/convert.py --hf-ckpt-path /raid/data/viraatc/models/deepseek-ai_DeepSeek-R1 --save-path /raid/data/viraatc/models/deepseek-ai_DeepSeek-R1-Demo --n-experts 256 --model-parallel 8

echo "Model download and conversion completed."

# Print setup information (with venv activated)
print_setup_info "$VENV_DIR" "pytorch" "true"

echo ""
echo "=== PyTorch Backend Setup Complete ==="
echo "Virtual environment is now active and will remain active."
echo "Ready for PyTorch distributed inference and MLPerf runs." 