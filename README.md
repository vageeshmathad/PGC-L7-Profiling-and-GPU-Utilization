# Experiment 7: Profiling and GPU Utilization

A PyTorch/CUDA experiment that profiles a batched GPU workload, identifies transfer/compute idle time, and compares a serial baseline with an overlapped implementation using pinned host memory, CUDA streams, and double buffering.

> **Results integrity:** The numerical results in the lab manual are reference/sample values, not measurements from this machine. This repository leaves the actual RTX 2050 measurements to be filled in after running the scripts.

## Objective

- Capture CPU and CUDA activity with `torch.profiler`.
- Export Chrome trace JSON files for Perfetto.
- Identify GPU kernels, memory copies, and idle gaps.
- Compare pageable/blocking transfers against pinned/non-blocking transfers.
- Overlap host-to-device copies with computation using two CUDA streams and two device buffers.
- Check correctness using matching checksums.

## Repository structure

```text
.
├── gpu_workload.py       # Baseline and optimized workloads
├── analyze_trace.py      # Timeline/idle-gap analyzer
├── requirements.txt
├── .gitignore
└── README.md
```

## Environment

Tested target for this lab: Windows, Python 3.12+, NVIDIA GPU, and a CUDA-enabled PyTorch build. The lab workstation used for this exercise is an NVIDIA GeForce RTX 2050 with 4 GB VRAM.

The CUDA version printed by `nvidia-smi` is the maximum CUDA version supported by the installed driver; it is not necessarily the CUDA toolkit installed locally. PyTorch wheels bundle their CUDA runtime.

### 1. Create and activate a virtual environment (Windows PowerShell)

```powershell
cd D:\Experiment7
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### 2. Install CUDA-enabled PyTorch

Use the PyTorch installation selector at https://pytorch.org/get-started/locally/ and choose Windows, Pip, Python, and a CUDA build compatible with your NVIDIA driver.

The lab manual's example command is:

```powershell
python -m pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Then verify:

```powershell
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Not detected'); print('PyTorch CUDA runtime:', torch.version.cuda)"
```

CUDA must report `True` before running the experiment.

## Run the experiment

Run commands from the directory containing both Python scripts.

### Baseline

Open a second terminal and log GPU utilization:

```powershell
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv -lms 100 > smi_baseline.csv
```

In the first terminal:

```powershell
python gpu_workload.py --mode baseline --profile
```

Stop the utilization logger with Ctrl+C after the workload finishes. The run writes `trace_baseline.json`.

### Analyze baseline

```powershell
python analyze_trace.py trace_baseline.json
```

### Optimized version

Start a second utilization logger:

```powershell
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv -lms 100 > smi_optimized.csv
```

Then run:

```powershell
python gpu_workload.py --mode optimized --profile
```

Stop the logger with Ctrl+C. The run writes `trace_optimized.json`.

### Compare traces

```powershell
python analyze_trace.py trace_baseline.json trace_optimized.json
```

Open each trace at https://ui.perfetto.dev (Open trace file) and capture a screenshot of the GPU copy and kernel tracks.

## Workload options

Default settings follow the lab manual: 200 batches, 16,384 rows, 1,024 columns, and 3 GEMM repetitions per batch.

For a lower-memory or quicker trial:

```powershell
python gpu_workload.py --mode baseline --rows 8192 --batches 20 --profile
python gpu_workload.py --mode optimized --rows 8192 --batches 20 --profile
```

Useful options:

| Option | Default | Meaning |
|---|---:|---|
| `--mode` | Required | `baseline` or `optimized` |
| `--batches` | 200 | Number of workload batches |
| `--rows` | 16384 | Rows per input batch |
| `--cols` | 1024 | Columns per input batch |
| `--reps` | 3 | GEMM/elementwise repetitions |
| `--profile` | Off | Export a CUDA/CPU profiler trace |
| `--profile-batches` | 10 | Batches included in the profiler trace |

## What to inspect

- **Baseline:** host-to-device copy, kernels, and device-to-host copy execute serially. GPU compute can be idle during transfers.
- **Optimized:** pinned host memory enables asynchronous transfers. A copy stream prefetches the next batch while a compute stream processes the current batch.
- **Double buffering:** two device buffers allow one buffer to be processed while the other is filled.
- **CUDA events:** `copied` prevents compute from reading an unfilled buffer; `freed` prevents the copy stream from overwriting a buffer still being read by compute.
- **Correctness:** baseline and optimized checksums should match (allowing tiny floating-point differences).

## Kernel classification

Arithmetic intensity (AI) is FLOPs divided by bytes moved. Compare AI with the GPU ridge point:

`ridge point = peak FP32 FLOP/s ÷ peak memory bandwidth (bytes/s)`

| Operation | Lab-manual AI | Expected classification |
|---|---:|---|
| GEMM | ~249 FLOP/byte | Compute-bound on the manual's example GPU |
| ReLU | 0.125 FLOP/byte | Memory-bound |
| Multiply / add | 0.125 FLOP/byte | Memory-bound |
| Sum reduction | 0.25 FLOP/byte | Memory-bound |

These are lab-manual estimates. Actual classification depends on the GPU and workload size. The analyzer's kernel-name label is only a heuristic; use arithmetic intensity and hardware specifications for the report.

## Results

Record your actual run values here after execution. Do not substitute the reference values below for measured results.

| Metric | Baseline | Optimized |
|---|---:|---:|
| GPU model | Pending | Pending |
| Total time (s) | Pending | Pending |
| Time per batch (ms) | Pending | Pending |
| Throughput (batches/s) | Pending | Pending |
| Average `nvidia-smi` GPU utilization (%) | Pending | Pending |
| Analyzer kernel busy / window (%) | Pending | Pending |
| Idle gaps > 50 µs (count / ms) | Pending | Pending |
| Copy overlapped with kernels (ms) | Pending | Pending |
| Checksum | Pending | Pending |

### Lab-manual reference results (not this machine's measurements)

The manual's illustrative mid-range GPU example reports 4.76 s baseline, 2.62 s optimized, and approximately 1.82x speedup. It shows utilization rising from about 55% to 96% in the `nvidia-smi` sample and kernel-busy/window rising from 52% to 94%. Your RTX 2050 results will differ.

## Deliverables checklist

- [ ] Before timeline screenshot (baseline)
- [ ] After timeline screenshot (optimized)
- [ ] Baseline and optimized utilization logs
- [ ] Table comparing runtime, throughput, and utilization
- [ ] Kernel classification with AI calculations and GPU ridge point
- [ ] Bottlenecks, evidence, and fixes
- [ ] Viva-question answers

## Troubleshooting

- **CUDA unavailable:** verify the NVIDIA driver and install a CUDA-enabled PyTorch wheel.
- **CUDA out of memory:** reduce `--rows` to 8192 or 4096 and/or reduce `--batches`.
- **No kernels in trace / CUPTI error:** close other profilers and Python GPU sessions, then retry.
- **No visible copy/compute overlap:** Windows WDDM or GPU/driver concurrency limitations may restrict overlap; report the observed timeline rather than assuming it worked.
- **Optimized run is not faster:** confirm the copy type is pinned, close GPU-heavy apps, use a plugged-in high-performance power mode, and compare the traces.

## Viva questions

1. Define GPU utilization. Why can `nvidia-smi` show 100% even when the GPU is not fully efficient?
2. Why is pinned memory faster than pageable memory for host-to-device transfers?
3. Why does overlapping copy and compute need two streams, and why are two device buffers needed?
4. Define arithmetic intensity and ridge point. Is GEMM always compute-bound? What if the matrix is tiny?
5. Why can measured speedup be less than the theoretical limit?
6. Why must the checksums of the two versions match?

## Source

Based on the KLE Technological University lab manual, **Experiment 7: Profiling and GPU Utilization**. The implementation and measurements should be validated on the target workstation before submission.
