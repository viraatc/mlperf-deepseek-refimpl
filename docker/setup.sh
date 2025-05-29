#!/bin/bash
# Main setup script for MLPerf DeepSeek evaluation environment
# This script detects the backend and calls the appropriate setup script
#
# Backend-specific behavior:
# - pytorch, vllm, sglang: Virtual environment activated after setup
# - trtllm: Virtual environment created but NOT activated (system Python for inference)

set -e  # Exit on error

# Function to detect backend based on installed packages and environment
detect_backend() {
    echo "Detecting backend environment..." >&2
    
    # Check for TensorRT-LLM (most specific first)
    if python3 -c "import tensorrt_llm" >/dev/null 2>&1; then
        echo "trtllm"
        return 0
    fi
    
    # Check for SGLang
    if python3 -c "import sglang" >/dev/null 2>&1; then
        echo "sglang"
        return 0
    fi
    
    # Check for vLLM
    if python3 -c "import vllm" >/dev/null 2>&1; then
        echo "vllm"
        return 0
    fi
    
    # Check for PyTorch with ref_dsinfer (DeepSeek-V3 reference implementation)
    if python3 -c "import torch" >/dev/null 2>&1 && python3 -c "import ref_dsinfer" >/dev/null 2>&1; then
        echo "pytorch"
        return 0
    fi
    
    # Check for general PyTorch environment (fallback)
    if python3 -c "import torch" >/dev/null 2>&1; then
        echo "pytorch"
        return 0
    fi
    
    # If no specific backend detected, default to a generic setup
    echo "unknown"
    return 1
}

# Parse command line arguments
FORCE_REBUILD=false
BACKEND_OVERRIDE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --force-rebuild)
            FORCE_REBUILD=true
            shift
            ;;
        --backend)
            BACKEND_OVERRIDE="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --force-rebuild    Force rebuild of MLPerf LoadGen from source"
            echo "  --backend BACKEND  Override backend detection (pytorch|vllm|sglang|trtllm)"
            echo "  --help            Show this help message"
            echo ""
            echo "This script automatically detects the backend and runs the appropriate setup."
            echo ""
            echo "Backend-specific behavior:"
            echo "  pytorch, vllm, sglang: Virtual environment activated after setup"
            echo "  trtllm: Virtual environment created but NOT activated"
            echo ""
            echo "For manual backend selection, use: --backend <backend_name>"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "=== MLPerf DeepSeek Environment Setup ==="

# Detect or use override backend
if [ -n "$BACKEND_OVERRIDE" ]; then
    BACKEND="$BACKEND_OVERRIDE"
    echo "Using backend override: $BACKEND"
else
    BACKEND=$(detect_backend)
    if [ $? -eq 0 ]; then
        echo "Detected backend: $BACKEND"
    else
        echo "Could not detect backend automatically."
        echo "Available backends: pytorch, vllm, sglang, trtllm"
        echo "Use --backend <backend> to specify manually."
        exit 1
    fi
fi

# Validate backend
case "$BACKEND" in
    pytorch|vllm|sglang|trtllm)
        echo "Setting up for $BACKEND backend..."
        ;;
    *)
        echo "Error: Unknown backend '$BACKEND'"
        echo "Supported backends: pytorch, vllm, sglang, trtllm"
        exit 1
        ;;
esac

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SETUP_SCRIPTS_DIR="$SCRIPT_DIR/setup_scripts"

# Check if setup scripts directory exists
if [ ! -d "$SETUP_SCRIPTS_DIR" ]; then
    echo "Error: Setup scripts directory not found at $SETUP_SCRIPTS_DIR"
    echo "Please ensure the setup_scripts directory exists and contains backend-specific scripts."
    exit 1
fi

# Call the appropriate backend setup script
BACKEND_SCRIPT="$SETUP_SCRIPTS_DIR/setup_$BACKEND.sh"

if [ ! -f "$BACKEND_SCRIPT" ]; then
    echo "Error: Backend setup script not found: $BACKEND_SCRIPT"
    echo "Available setup scripts:"
    ls -1 "$SETUP_SCRIPTS_DIR"/setup_*.sh 2>/dev/null || echo "  None found"
    exit 1
fi

echo "Running backend-specific setup: $BACKEND_SCRIPT"
echo ""

# Make sure the script is executable (only if we have permission to change it)
if [ ! -x "$BACKEND_SCRIPT" ]; then
    if chmod +x "$BACKEND_SCRIPT" 2>/dev/null; then
        echo "Made script executable: $BACKEND_SCRIPT"
    else
        echo "Note: Could not make script executable, but will run with bash"
    fi
fi

# Pass through the force rebuild flag if set
if [ "$FORCE_REBUILD" = "true" ]; then
    bash "$BACKEND_SCRIPT" --force-rebuild
else
    bash "$BACKEND_SCRIPT"
fi

echo ""
echo "=== Main Setup Complete ==="
echo "Backend: $BACKEND"

# Provide activation instructions for backends that use virtual environments
case "$BACKEND" in
    pytorch|vllm|sglang)
        echo ""
        echo "IMPORTANT: To activate the virtual environment, run:"
        echo "   source /work/.venv_${BACKEND}/bin/activate"
        echo ""
        echo ""
        if [ "$BACKEND" = "pytorch" ]; then
            echo "   (venv) torchrun --nproc_per_node=8 run_eval_mpi.py ..."
            echo "   (venv) torchrun --nproc_per_node=8 run_mlperf_mpi.py ..."
        else
            echo "   (venv) python run_eval.py --backend $BACKEND ..."
            echo "   (venv) python run_mlperf.py --backend $BACKEND ..."
        fi
        ;;
    trtllm)
        echo ""
        echo "For TensorRT-LLM backend, use system Python (do NOT activate venv for inference):"
        echo "   python run_eval.py --backend trtllm ..."
        echo "   python run_mlperf.py --backend trtllm ..."
        echo ""
        echo "Only activate venv for accuracy evaluation:"
        echo "   source /work/.venv_trtllm/bin/activate"
        echo "   python eval_accuracy.py ..."
        ;;
esac

echo ""
echo "For usage instructions, see the output above or run:"
echo "  $BACKEND_SCRIPT --help" 