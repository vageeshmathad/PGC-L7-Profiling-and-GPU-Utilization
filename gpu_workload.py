"""Experiment 7: Profiling and GPU Utilization.

Baseline: pageable host memory, default stream, blocking copies.
Optimized: pinned host memory, separate copy/compute streams, double buffering.
"""
import argparse
import math
import time

import torch


def make_weights(n: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(1)
    return (torch.randn(n, n, generator=generator) / math.sqrt(n)).cuda()


def make_host_batches(rows: int, cols: int, pinned: bool, unique: int = 4):
    """Create a small reusable set of host batches to limit host RAM use."""
    generator = torch.Generator().manual_seed(0)
    batches = []
    for _ in range(unique):
        tensor = torch.randn(rows, cols, generator=generator)
        batches.append(tensor.pin_memory() if pinned else tensor)
    return batches


def compute(x: torch.Tensor, w: torch.Tensor, reps: int) -> torch.Tensor:
    """GPU work for one batch: GEMM, elementwise operations, then reduction."""
    y = x
    for _ in range(reps):
        y = y @ w
        y = torch.relu(y)
        y = y.mul_(0.5).add_(1.0)
    return y.sum(dim=1)


def run_baseline(host, w, reps: int, nbatches: int) -> torch.Tensor:
    results = []
    for i in range(nbatches):
        x = host[i % len(host)].cuda()
        result = compute(x, w, reps)
        results.append(result.cpu())
    torch.cuda.synchronize()
    return torch.stack(results)


def run_optimized(host, w, reps: int, nbatches: int) -> torch.Tensor:
    rows, cols = host[0].shape
    copy_stream = torch.cuda.Stream()
    compute_stream = torch.cuda.Stream()

    buffers = [torch.empty(rows, cols, device="cuda") for _ in range(2)]
    copied = [torch.cuda.Event() for _ in range(2)]
    freed = [torch.cuda.Event() for _ in range(2)]
    output_host = torch.empty(nbatches, rows, pin_memory=True)

    torch.cuda.synchronize()
    for event in freed:
        event.record(compute_stream)

    def issue_copy(batch_index: int) -> None:
        slot = batch_index % 2
        copy_stream.wait_event(freed[slot])
        with torch.cuda.stream(copy_stream):
            buffers[slot].copy_(
                host[batch_index % len(host)], non_blocking=True
            )
            copied[slot].record(copy_stream)

    issue_copy(0)
    for i in range(nbatches):
        slot = i % 2
        if i + 1 < nbatches:
            issue_copy(i + 1)

        compute_stream.wait_event(copied[slot])
        with torch.cuda.stream(compute_stream):
            result = compute(buffers[slot], w, reps)
            freed[slot].record(compute_stream)
            output_host[i].copy_(result, non_blocking=True)

    torch.cuda.synchronize()
    return output_host


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("baseline", "optimized"), required=True)
    parser.add_argument("--batches", type=int, default=200)
    parser.add_argument("--rows", type=int, default=16384)
    parser.add_argument("--cols", type=int, default=1024)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-batches", type=int, default=10)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU not available - check driver / CUDA-enabled PyTorch.")
    if min(args.batches, args.rows, args.cols, args.reps, args.profile_batches) < 1:
        raise SystemExit("batches, rows, cols, reps, and profile-batches must be positive.")

    pinned = args.mode == "optimized"
    workload = run_optimized if pinned else run_baseline
    host = make_host_batches(args.rows, args.cols, pinned)
    weights = make_weights(args.cols)

    print(f"GPU             : {torch.cuda.get_device_name(0)}")
    print(f"PyTorch         : {torch.__version__}")
    print(f"PyTorch CUDA    : {torch.version.cuda}")
    print(f"Mode            : {args.mode}")
    print(f"Batch size      : {args.rows * args.cols * 4 / 1e6:.1f} MB")

    workload(host, weights, args.reps, 3)
    torch.cuda.synchronize()

    start = time.perf_counter()
    results = workload(host, weights, args.reps, args.batches)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    print(f"Batches         : {args.batches}")
    print(f"Total time      : {elapsed:.3f} s")
    print(f"Per batch       : {elapsed / args.batches * 1000:.2f} ms")
    print(f"Throughput      : {args.batches / elapsed:.1f} batches/s")
    print(f"Checksum        : {results.double().sum().item():.6e}")

    if args.profile:
        from torch.profiler import ProfilerActivity, profile

        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]
        ) as profiler:
            workload(host, weights, args.reps, args.profile_batches)
            torch.cuda.synchronize()

        trace_name = f"trace_{args.mode}.json"
        profiler.export_chrome_trace(trace_name)
        print(f"\nTrace saved     : {trace_name}")
        for sort_key in ("cuda_time_total", "device_time_total"):
            try:
                print(profiler.key_averages().table(sort_by=sort_key, row_limit=10))
                break
            except Exception:
                continue


if __name__ == "__main__":
    main()
