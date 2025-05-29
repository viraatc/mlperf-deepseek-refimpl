# Mlperf Inference DeepSeek Reference Implementation

## Docker Launch System

The MLPerf DeepSeek reference implementation includes a comprehensive Docker launch system that supports multiple backends and provides advanced features like user management, persistent storage, and flexible configuration.

### Quick Start

Launch a Docker container with your preferred backend:

```bash
# Launch PyTorch backend
./launch.sh --backend pytorch

# Launch vLLM backend
./launch.sh --backend vllm

# Launch SGLang backend
./launch.sh --backend sglang

# See launch.sh for full list of args
./launch.sh --backend vllm --gpu-count 2 --extra-mounts "/data:/data,/models:/models" --local-user 0
```

### Available Backends

- **pytorch**: PyTorch-based inference with DeepSeek-V3 support via `ref_dsinfer` package
- **vllm**: vLLM high-performance inference engine
- **sglang**: SGLang serving framework

## Backend-Specific Setup

After launching any Docker container, run the setup script which automatically detects your backend:

```bash
# Automatic backend detection and setup
setup.sh
```

The setup script creates a virtual environment and configures it differently based on the backend:

#### All Backends
- Virtual environment is **activated** after setup
- All commands are to be run using the virtual environment

## Running Evaluations

### PyTorch Backend (Distributed)

PyTorch backend uses distributed execution with `torchrun`:

```bash
# Regular inference evaluation
(venv) $ torchrun --nproc_per_node=8 run_eval_mpi.py --input-file data/final_output.pkl --output-file data/pytorch_output.pkl --num-samples 32

# MLPerf performance benchmarks
(venv) $ torchrun --nproc_per_node=8 run_mlperf_mpi.py --mode offline --input-file data/final_output.pkl --output-dir mlperf_results

# MLPerf accuracy mode 
(venv) $ torchrun --nproc_per_node=8 run_mlperf_mpi.py --mode offline --accuracy --input-file data/final_output.pkl --output-dir mlperf_results
```

### vLLM and SGLang Backends

For vLLM and SGLang, use single-process execution:

```bash
# Regular inference evaluation
(venv) $ python run_eval.py --backend vllm --input-file data/final_output.pkl
(venv) $ python run_eval.py --backend sglang --input-file data/final_output.pkl --async

# MLPerf performance benchmarks  
(venv) $ python run_mlperf.py --backend vllm --mode offline --input-file data/final_output.pkl --output-dir mlperf_results
(venv) $ python run_mlperf.py --backend sglang --mode server --input-file data/final_output.pkl --output-dir mlperf_results
```

## MLPerf Inference Support

The reference implementation includes full support for MLPerf inference benchmarks through a System Under Test (SUT) wrapper that integrates with MLPerf LoadGen.

### Running MLPerf Benchmarks

#### Offline Scenario
```bash
# vLLM backend
(venv) $ python run_mlperf.py \
    --backend vllm \
    --mode offline \
    --input-file data/final_output.pkl \
    --output-dir mlperf_results

# SGLang backend
(venv) $ python run_mlperf.py \
    --backend sglang \
    --mode offline \
    --input-file data/final_output.pkl \
    --output-dir mlperf_results
```

#### Server Scenario
```bash
# SGLang backend
(venv) $ python run_mlperf.py \
    --backend sglang \
    --mode server \
    --input-file data/final_output.pkl \
    --output-dir mlperf_results
```

#### PyTorch Distributed MLPerf
```bash
# PyTorch MLPerf offline scenario
(venv) $ torchrun --nproc_per_node=8 run_mlperf_mpi.py \
    --mode offline \
    --input-file data/final_output.pkl \
    --output-dir mlperf_results
```

### MLPerf Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--backend` | Backend to use (required) | - |
| `--mode` | Scenario mode (offline/server) | `offline` |
| `--accuracy` | Run accuracy test | `False` |
| `--output-dir` | Output directory for results | `mlperf_results` |

### Configuration Files

The MLPerf configuration files are located in the `mlperf/` directory:
- `mlperf/mlperf.conf`: Base MLPerf configuration
- `mlperf/user.conf`: User-specific overrides

## Accuracy Evaluation

Accuracy evaluation is handled uniformly across all backends:

```bash
# within container, with virtualenv activated
(venv) $ python3 eval_accuracy.py --input-file input_file.pkl 
```

## Summary of Backend Usage Patterns

| Backend | Virtual Environment | Regular Commands | Accuracy Evaluation |
|---------|-------------------|------------------|-------------------|
| **PyTorch** | Use Always | `torchrun --nproc_per_node=8 run_eval_mpi.py ...` | `python eval_accuracy.py ...` |
| **vLLM** | Use Always | `python run_eval.py --backend vllm ...` | `python eval_accuracy.py ...` |
| **SGLang** | Use Always | `python run_eval.py --backend sglang ...` | `python eval_accuracy.py ...` |
