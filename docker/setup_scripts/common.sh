#!/bin/bash
# Common setup functions for MLPerf DeepSeek evaluation environment
# This script contains shared functionality for all backends

set -e  # Exit on error

# Common function to check if uv is installed
check_uv_installed() {
    if ! command -v uv &> /dev/null; then
        echo "Error: uv is not installed. Please ensure the Docker image has uv installed."
        exit 1
    fi
}

# Common function to setup virtual environment
setup_virtual_environment() {
    local VENV_DIR="$1"
    
    # Check if we're already in the /work directory
    if [ "$PWD" != "/work" ]; then
        echo "Changing to /work directory..."
        cd /work
    fi

    # Check if virtual environment already exists
    if [ -d "$VENV_DIR" ]; then
        echo "Virtual environment already exists at $VENV_DIR"
    else
        echo "Creating new UV virtual environment..."
        uv venv --system-site-packages "$VENV_DIR"
    fi
}

# Common function to install build dependencies
install_build_dependencies() {
    echo "Installing build dependencies in venv..."
    uv pip install numpy setuptools
}

# Common function to patch prm800 setup.py
patch_prm800k_setup() {
    echo "Checking and patching prm800 setup.py if necessary..."
    PRM800_SETUP="/work/submodules/prm800/setup.py"
    if [ -f "$PRM800_SETUP" ]; then
        echo "Found prm800 setup.py at: $PRM800_SETUP"
        # Check if the file still has the problematic import
        if grep -q "import numpy" "$PRM800_SETUP"; then
            echo "Patching prm800 setup.py to fix numpy import issue..."
            # Create a backup
            cp "$PRM800_SETUP" "$PRM800_SETUP.bak"
            # Remove the numpy import  line and add it to install_requires
            cat > "$PRM800_SETUP" << 'EOF'
from setuptools import setup, find_packages

setup(
    name='prm800',
    packages=find_packages(),
    version='0.0.1',
    install_requires=[
        'numpy',
    ],
)
EOF
            echo "Patch applied successfully!"
            echo "New content of setup.py:"
            cat "$PRM800_SETUP"
        else
            echo "prm800 setup.py already patched or doesn't need patching."
        fi
    else
        echo "WARNING: prm800 setup.py not found at $PRM800_SETUP"
    fi
}

# Common function to install evaluation requirements
install_evaluation_requirements() {
    echo "Installing evaluation requirements..."
    if [ -f "/work/docker/evaluation_requirements.txt" ]; then
        uv pip install -r /work/docker/evaluation_requirements.txt
        echo "Evaluation requirements installed successfully!"
    else
        echo "Warning: evaluation_requirements.txt not found at /work/docker/evaluation_requirements.txt"
        echo "Please ensure the workspace is properly mounted."
    fi
}

# Common function to check build tools
check_build_tools() {
    echo "Checking if required build tools are available..."
    for tool in cmake make git g++; do
        if ! command -v $tool &> /dev/null; then
            echo "Error: $tool is not installed. MLPerf LoadGen installation cannot proceed."
            echo "Please ensure build dependencies are installed in the Docker image."
            exit 1
        fi
    done
}

# Common function to install MLPerf LoadGen
install_mlperf_loadgen() {
    local FORCE_REBUILD="$1"
    
    echo ""
    echo "=== Installing MLPerf LoadGen ==="
    
    check_build_tools
    
    # Set build directory
    BUILD_DIR="/work/build"
    MLPERF_BUILD_DIR="$BUILD_DIR/mlperf_loadgen"
    WHEEL_DIR="$BUILD_DIR/wheels"

    # Create build directories
    mkdir -p $MLPERF_BUILD_DIR
    mkdir -p $WHEEL_DIR

    # Check if force rebuild is requested
    if [ "$FORCE_REBUILD" = "true" ]; then
        echo "Force rebuild requested. Removing cached wheels..."
        rm -f $WHEEL_DIR/mlcommons_loadgen*.whl
    fi

    # Check if we already have a built wheel
    if ls $WHEEL_DIR/mlcommons_loadgen*.whl 1> /dev/null 2>&1; then
        echo "Found existing MLPerf LoadGen wheel in $WHEEL_DIR"
        
        # Get Python version info for compatibility check
        PYTHON_VERSION=$(python3 -c "import sys; print(f'cp{sys.version_info.major}{sys.version_info.minor}')")
        PLATFORM=$(python3 -c "import platform; print(platform.machine())")
        
        # Check if the wheel is compatible with current Python version
        WHEEL_FILE=$(ls $WHEEL_DIR/mlcommons_loadgen*.whl | head -n1)
        WHEEL_NAME=$(basename "$WHEEL_FILE")
        
        echo "Current Python version tag: $PYTHON_VERSION"
        echo "Current platform: $PLATFORM"
        echo "Found wheel: $WHEEL_NAME"
        
        # Check if wheel matches current Python version and platform
        if [[ "$WHEEL_NAME" == *"$PYTHON_VERSION"* ]] && [[ "$WHEEL_NAME" == *"$PLATFORM"* ]]; then
            echo "Wheel is compatible. Installing from cached wheel..."
            
            # install to system python only
            pip install --force-reinstall $WHEEL_FILE
            
            # Verify installation
            if python3 -c "import mlperf_loadgen" 2>/dev/null; then
                echo "MLPerf LoadGen installed successfully from cached wheel!"
                return 0
            else
                echo "Installation verification failed. Will rebuild..."
                rm -f $WHEEL_DIR/mlcommons_loadgen*.whl
                NEED_BUILD=true
            fi
        else
            echo "Cached wheel is incompatible with current environment."
            echo "Removing old wheel and rebuilding..."
            rm -f $WHEEL_DIR/mlcommons_loadgen*.whl
            NEED_BUILD=true
        fi
    else
        echo "No cached wheel found."
        NEED_BUILD=true
    fi

    # Build from source if needed
    if [ "${NEED_BUILD:-true}" = "true" ]; then
        echo "Building MLPerf LoadGen from source..."
        
        cd $MLPERF_BUILD_DIR
        
        # Clone or update MLPerf inference repository
        if [ -d "inference" ]; then
            echo "Updating existing MLPerf inference repository..."
            cd inference
            git pull
            git submodule update --init --recursive
        else
            echo "Cloning MLPerf inference repository..."
            git clone https://github.com/mlcommons/inference.git
            cd inference
            git submodule update --init --recursive
        fi
        
        # Navigate to loadgen directory
        LOADGEN_DIR=$MLPERF_BUILD_DIR/inference/loadgen
        cd $LOADGEN_DIR
        
        # Build loadgen C++ library first
        echo "Building loadgen C++ library..."
        mkdir -p build
        cd build
        cmake -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DCMAKE_BUILD_TYPE=Release ..
        make -j$(nproc)
        
        # Go back to loadgen directory to build wheel
        cd $LOADGEN_DIR
        
        # Build the wheel using UV to avoid dependency conflicts
        echo "Building MLPerf LoadGen wheel..."
        # First, ensure we have the build dependencies
        uv pip install setuptools wheel "pybind11>=2.11.1"
        
        # Build the wheel using python setup.py directly to avoid pip's dependency resolver conflicts
        CFLAGS="-std=c++14 -O3" python3 setup.py bdist_wheel --dist-dir=$WHEEL_DIR
        
        # Install the wheel using pip directly
        echo "Installing MLPerf LoadGen wheel..."
        WHEEL_FILE=$(ls $WHEEL_DIR/mlcommons_loadgen*.whl | head -n1)
        pip install --force-reinstall $WHEEL_FILE
        
        # Verify installation
        if python3 -c "import mlperf_loadgen" 2>/dev/null; then
            echo "MLPerf LoadGen built and installed successfully!"
        else
            echo "Error: MLPerf LoadGen installation verification failed!"
            exit 1
        fi
    fi

    # Return to work directory
    cd /work
}

# Common function to print final setup info
print_setup_info() {
    local VENV_DIR="$1"
    local BACKEND="$2"
    local ACTIVATE_VENV="$3"
    
    echo ""
    echo "=== Setup completed successfully ==="
    echo "Virtual environment created at: $VENV_DIR"
    echo "Backend: $BACKEND"
    echo ""
    echo "IMPORTANT USAGE NOTES:"
    echo "====================="
    
    if [ "$ACTIVATE_VENV" = "true" ]; then
        echo "Virtual environment has been activated for this backend."
        echo ""
        echo "For all commands with this backend, the virtual environment is active:"
        case "$BACKEND" in
            "pytorch")
                echo "   (venv) \$ torchrun --nproc_per_node=8 run_eval_mpi.py ..."
                echo "   (venv) \$ torchrun --nproc_per_node=8 run_mlperf_mpi.py ..."
                ;;
            "vllm")
                echo "   (venv) \$ python run_eval.py --backend vllm ..."
                echo "   (venv) \$ python run_mlperf.py --backend vllm ..."
                ;;
            "sglang")
                echo "   (venv) \$ python run_eval.py --backend sglang ..."
                echo "   (venv) \$ python run_mlperf.py --backend sglang ..."
                ;;
        esac
        echo "   \$ python eval_accuracy.py ..."
    else
        echo "Virtual environment NOT activated for this backend ($BACKEND)."
        echo ""
        echo "For regular inference and MLPerf runs with this backend:"
        echo "   DO NOT activate the virtual environment. Run directly:"
        case "$BACKEND" in
            "trtllm")
                echo "   \$ python run_eval.py --backend trtllm ..."
                echo "   \$ python run_mlperf.py --backend trtllm ..."
                ;;
        esac
        echo ""
        echo "For accuracy evaluation ONLY, activate the virtual environment:"
        echo "   \$ source $VENV_DIR/bin/activate"
        echo "   (venv) \$ python eval_accuracy.py ..."
        echo "   (venv) \$ deactivate"
    fi
    
    echo ""
    echo "MLPerf LoadGen installed as a system Python package (available everywhere)"
    echo "Build artifacts cached at: /work/build"
    echo "Wheels cached at: /work/build/wheels"
} 