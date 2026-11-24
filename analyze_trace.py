"""Analyze PyTorch Chrome traces for GPU kernel activity and idle gaps.

Usage:
    python analyze_trace.py trace_baseline.json trace_optimized.json
"""
import collections
import json
import sys

GAP_US = 50
COMPUTE_HINTS = ("gemm", "sgemm", "cutlass", "cublas", "conv", "wmma")


def merge(intervals):
    intervals = sorted(intervals)
    merged = []
    for start, end in intervals:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


def total(intervals):
    return sum(end - start for start, end in intervals)


def overlap(first, second):
    i = j = duration = 0
    while i < len(first) and j < len(second):
        start = max(first[i][0], second[j][0])
        end = min(first[i][1], second[j][1])
        if end > start:
            duration += end - start
        if first[i][1] < second[j][1]:
            i += 1
        else:
            j += 1
    return duration


def label_kernel(name):
    lowered = name.lower()
    return "compute-bound" if any(hint in lowered for hint in COMPUTE_HINTS) else "memory-bound"


def analyze(path):
    with open(path, encoding="utf-8") as trace_file:
        data = json.load(trace_file)
    events = data["traceEvents"] if isinstance(data, dict) else data

    kernels, copies = [], []
    kernel_names = collections.defaultdict(lambda: [0, 0.0])
    copy_names = collections.defaultdict(lambda: [0, 0.0])

    for event in events:
        if event.get("ph") != "X":
            continue
        category = event.get("cat", "")
        start = event.get("ts")
        duration = event.get("dur", 0)
        if start is None:
            continue
        end = start + duration

        if category == "kernel":
            kernels.append([start, end])
            kernel_names[event.get("name", "unknown")[:60]][0] += 1
            kernel_names[event.get("name", "unknown")[:60]][1] += duration
        elif category in ("gpu_memcpy", "gpu_memset"):
            copies.append([start, end])
            copy_names[event.get("name", "unknown")[:60]][0] += 1
            copy_names[event.get("name", "unknown")[:60]][1] += duration

    if not kernels:
        print(f"{path}: no GPU kernels found in trace")
        return None

    kernels = merge(kernels)
    copies = merge(copies)
    start = min(kernels[0][0], copies[0][0] if copies else kernels[0][0])
    end = max(kernels[-1][1], copies[-1][1] if copies else kernels[-1][1])
    span = end - start
    kernel_busy = total(kernels)
    gaps = [
        kernels[index + 1][0] - kernels[index][1]
        for index in range(len(kernels) - 1)
    ]
    large_gaps = [gap for gap in gaps if gap > GAP_US]

    print("=" * 72)
    print(path)
    print("=" * 72)
    print(f"Profiled window            : {span / 1000:9.2f} ms")
    print(f"Kernel busy time           : {kernel_busy / 1000:9.2f} ms")
    print(f"GPU utilization (busy/span): {100 * kernel_busy / span:9.1f} %")
    print(f"Copy (memcpy) busy time    : {total(copies) / 1000:9.2f} ms")
    print(f"Copy overlapped with kernel: {overlap(kernels, copies) / 1000:9.2f} ms")
    print(
        f"Idle gaps > {GAP_US} us        : {len(large_gaps):9d}   "
        f"(total {sum(large_gaps) / 1000:.2f} ms)"
    )

    print("\nKernels (name-based hint only; validate with arithmetic intensity)")
    print(f"{'calls':>6} {'total ms':>10} {'label':<14} name")
    for name, (count, duration) in sorted(
        kernel_names.items(), key=lambda item: -item[1][1]
    ):
        print(f"{count:6d} {duration / 1000:10.2f} {label_kernel(name):<14} {name}")

    print("\nMemory copies")
    print(f"{'calls':>6} {'total ms':>10} name")
    for name, (count, duration) in sorted(
        copy_names.items(), key=lambda item: -item[1][1]
    ):
        print(f"{count:6d} {duration / 1000:10.2f} {name}")
    print()

    return span, kernel_busy, total(copies), overlap(kernels, copies), len(large_gaps)


def main():
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python analyze_trace.py trace_*.json")

    rows = []
    for trace_path in sys.argv[1:]:
        result = analyze(trace_path)
        if result:
            rows.append((trace_path, result))

    if len(rows) > 1:
        print("SUMMARY")
        print(f"{'file':<28}{'window ms':>11}{'util %':>9}{'overlap ms':>13}{'gaps':>7}")
        for path, (span, busy, _copy, overlap_time, gap_count) in rows:
            utilization = 100 * busy / span if span else 0.0
            print(
                f"{path:<28}{span / 1000:11.1f}{utilization:9.1f}"
                f"{overlap_time / 1000:13.1f}{gap_count:7d}"
            )


if __name__ == "__main__":
    main()
