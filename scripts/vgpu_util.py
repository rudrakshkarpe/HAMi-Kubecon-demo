#!/usr/bin/env python3
"""Pretty-print HAMi per-vGPU utilization + memory from the device-plugin
monitor metrics endpoint.

Usage:
  python3 vgpu_util.py [METRICS_URL]
  (default URL: http://localhost:31992/metrics)

This shows what `nvidia-smi` inside a pod cannot: compute (SM) utilization and
memory attributed to each individual vGPU / container sharing one physical GPU.
"""
import re
import sys
import urllib.request

URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:31992/metrics"

# minimal ANSI colors (skip if not a tty)
T = sys.stdout.isatty()
BOLD = "\033[1m" if T else ""
DIM = "\033[2m" if T else ""
RST = "\033[0m" if T else ""
CYAN = "\033[36m" if T else ""
GREEN = "\033[32m" if T else ""
YEL = "\033[33m" if T else ""


def label(s, key):
    m = re.search(key + r'="([^"]*)"', s)
    return m.group(1) if m else ""


def main():
    try:
        raw = urllib.request.urlopen(URL, timeout=5).read().decode()
    except Exception as e:  # noqa
        print(f"  (could not reach HAMi monitor at {URL}: {e})")
        return 1

    util, used, limit = {}, {}, {}
    host_util, host_mem = None, None
    for line in raw.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        try:
            metric, value = line.rsplit(" ", 1)
            value = float(value)
        except ValueError:
            continue
        if metric.startswith("hami_container_device_utilization_ratio"):
            key = (label(metric, "pod"), label(metric, "container"))
            util[key] = value
        elif metric.startswith("hami_vgpu_memory_used_bytes"):
            key = (label(metric, "pod"), label(metric, "container"))
            used[key] = value
        elif metric.startswith("hami_vgpu_memory_limit_bytes"):
            key = (label(metric, "pod"), label(metric, "container"))
            limit[key] = value
        elif metric.startswith("hami_host_gpu_utilization_ratio"):
            host_util = value
        elif metric.startswith("hami_host_gpu_memory_used_bytes"):
            host_mem = value

    keys = sorted(set(util) | set(used) | set(limit))
    if not keys:
        print("  (no vGPU containers reporting yet)")
        return 0

    gb = lambda b: b / (1024 ** 3)  # noqa: E731
    bar = lambda pct: ("█" * int(round(pct / 5))).ljust(20)  # noqa: E731

    print(f"  {BOLD}{'POD / CONTAINER':<42}{'GPU-MEM (used/limit)':>22}{'SM-UTIL':>10}{RST}")
    print(f"  {DIM}{'-' * 84}{RST}")
    for k in keys:
        pod, ctr = k
        u = util.get(k, 0.0)
        mem_used = gb(used.get(k, 0.0))
        mem_lim = gb(limit.get(k, 0.0))
        name = f"{pod}/{ctr}"
        if len(name) > 41:
            name = name[:38] + "..."
        color = GREEN if u >= 1 else DIM
        print(f"  {CYAN}{name:<42}{RST}{mem_used:>9.1f}/{mem_lim:<5.1f} GB"
              f"{color}{u:>8.0f}%{RST}  {color}{bar(u)}{RST}")

    print(f"  {DIM}{'-' * 84}{RST}")
    if host_util is not None:
        hm = f"{gb(host_mem):.1f} GB" if host_mem is not None else "?"
        print(f"  {YEL}{BOLD}PHYSICAL H100{RST}{YEL}  → real SM util: {host_util:.0f}%   "
              f"real mem used: {hm}{RST}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
