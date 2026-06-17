# HAMi GPU Virtualization + KitOps/Jozu Hub Inference Demo

End-to-end demo of **GPU virtualization with [HAMi](https://github.com/Project-HAMi/HAMi)**
on a single NVIDIA **H100 80GB**, serving an LLM with **vLLM** and **SGLang**,
where the model is delivered from the **[Jozu Hub](https://jozu.ml) registry**
as a **KitOps ModelKit** instead of a runtime Hugging Face download.

It shows three things at once:

1. **HAMi** slicing one physical H100 into capped, schedulable vGPUs (memory + compute).
2. **KitOps / Jozu Hub** as the model supply chain — model packaged as an OCI artifact.
3. **vLLM** and **SGLang** serving that model, two different ways to consume it.

---

## Table of contents

1. [Architecture](#architecture)
2. [Environment](#environment)
3. [Cluster + HAMi (already provisioned)](#cluster--hami)
4. [The model on Jozu Hub](#the-model-on-jozu-hub)
5. [Two ways to consume the model](#two-ways-to-consume-the-model)
   - [Mode A — ModelKit + unpack + custom command](#mode-a--modelkit--unpack--custom-command)
   - [Mode B — Jozu RIC (turnkey image)](#mode-b--jozu-ric-turnkey-image)
6. [HAMi GPU virtualization & caps](#hami-gpu-virtualization--caps)
   - [Capping a workload to 60% of the GPU](#capping-a-workload-to-60-of-the-gpu)
7. [Build & deploy](#build--deploy)
8. [Running the guided demo](#running-the-guided-demo)
9. [Inference examples](#inference-examples)
10. [File map](#file-map)
11. [Troubleshooting](#troubleshooting)

---

## Architecture

```
                       Jozu Hub  (jozu.ml, OCI registry, anonymous pull)
                       ├── ModelKit:  jonathangamer202002/qwen3-4b-instruct:latest      (data: safetensors)
                       └── RIC:        jonathangamer202002/qwen3-4b-instruct/vllm:latest (runnable: vLLM + model)
                                          │
                ┌─────────────────────────┴──────────────────────────┐
        Mode A  │  ModelKit pulled + unpacked          Mode B  │  RIC pulled as one image
                ▼                                              ▼
┌───────────────────────────────────────┐      ┌───────────────────────────────────────┐
│ Pod (hami-scheduler)                   │      │ Pod (hami-scheduler)                   │
│  initContainer: hami-kitunpacker       │      │  container: jozu .../vllm:latest       │
│    kit unpack -> flatten -> /models    │      │    entrypoint serves vllm on :8000     │
│  container: hami-vllm-jozu / sglang    │      │    (model baked into the image)        │
│    custom cmd: serve /models/qwen3     │      │                                        │
│  HAMi caps: gpumem + gpucores          │      │  HAMi caps: gpumem + gpucores          │
└───────────────────────────────────────┘      └───────────────────────────────────────┘
                          \                      /
                           ▼                    ▼
                    ┌──────────────────────────────────┐
                    │  1× NVIDIA H100 80GB (kind node)  │
                    │  HAMi virtualizes -> vGPU slices  │
                    └──────────────────────────────────┘
```

---

## Environment

| Component | Version / value |
| --- | --- |
| GPU | NVIDIA H100 80GB HBM3 (`81559 MiB`) |
| Kubernetes | `kind` cluster `hami-demo`, node `hami-demo-control-plane` |
| HAMi | core `v2.9.0` (scheduler + device plugin) |
| NVIDIA GPU Operator | `v26.3.1` |
| vLLM image | `vllm/vllm-openai:latest` |
| SGLang image | `lmsysorg/sglang:latest` |
| KitOps CLI | `kit` `v1.14.0` |
| Model | Qwen3-4B-Instruct-2507 (safetensors, `Qwen3ForCausalLM`) |

CLI tools used: `kubectl`, `kind`, `helm`, `helmfile`, `kit`, `docker` (via `sg docker -c '…'`).

---

## Cluster + HAMi

The kind cluster and HAMi core were already provisioned (see `kind-gpu.yaml`,
`helmfile.d/01-hami-core.yaml`). Sanity checks:

```bash
kubectl get nodes
kubectl -n kube-system get pods | grep -E 'hami|device-plugin|scheduler'

# HAMi advertises the single H100 as multiple schedulable vGPUs:
kubectl get node hami-demo-control-plane -o jsonpath='{.status.capacity.nvidia\.com/gpu}{"\n"}'
```

HAMi resource knobs used throughout this demo:

| Resource | Meaning |
| --- | --- |
| `nvidia.com/gpu` | number of vGPUs to request |
| `nvidia.com/gpumem` | VRAM cap in **MB** (absolute) |
| `nvidia.com/gpumem-percentage` | VRAM cap as **% of the device** |
| `nvidia.com/gpucores` | compute (SM) cap as **% (0–100)** |

A pod opts into HAMi by setting `schedulerName: hami-scheduler`.

---

## The model on Jozu Hub

The model is a **public, anonymously-pullable** KitOps ModelKit:

```
jozu.ml/jonathangamer202002/qwen3-4b-instruct:latest
```

Inspect it without logging in:

```bash
kit inspect --remote jozu.ml/jonathangamer202002/qwen3-4b-instruct:latest
```

It contains a Hugging Face safetensors model (`config.json`, `tokenizer.json`,
`model.safetensors.index.json`, and 3 shards) — so both vLLM and SGLang can serve it.

> **Layout quirk:** this ModelKit stores the `.safetensors` shards in a `model/`
> **subdirectory** while `config.json` / the index sit one level up. vLLM/transformers
> need them in **one** directory, so both modes below **flatten** the layout.

---

## Two ways to consume the model

Jozu Hub exposes the same model in two forms; this repo implements both.

| | Mode A — ModelKit + unpack | Mode B — RIC |
| --- | --- | --- |
| Ref | `…/qwen3-4b-instruct:latest` (data) | `…/qwen3-4b-instruct/vllm:latest` (image) |
| Model delivery | `kitunpacker` initContainer pulls + unpacks | baked into the image |
| Serve command | **custom**, baked into our Dockerfile | the RIC's built-in entrypoint |
| Engines | vLLM **and** SGLang (shared model cache) | vLLM |
| Manifests | `kitops/charts/vllm-jozu.yaml`, `kitops/charts/sglang-jozu.yaml` | `kitops/charts/vllm-ric-jozu.yaml` |

### Mode A — ModelKit + unpack + custom command

The `kitunpacker` init container (`kitops/docker/kitunpacker/`) pulls the ModelKit
from Jozu Hub, unpacks the `model` layers, and **flattens** them into
`/models/qwen3` on a shared node-local volume (`hostPath: /var/lib/hami-demo/modelkit`,
shared by both engines with a lock + skip-if-cached so the 8 GB pull happens once).

The engine images (`kitops/docker/vllm/`, `kitops/docker/sglang/`) bake a **custom
serve command** that loads the **local** model — no Hugging Face download:

```bash
# vLLM (custom):   vllm serve /models/qwen3   --served-model-name qwen3-4b-instruct ...
# SGLang (custom): python3 -m sglang.launch_server --model-path /models/qwen3 ...
```

vs. the default HF flow we replaced (`vllm serve Qwen/Qwen3-1.7B`).

Served model name: **`qwen3-4b-instruct`** · vLLM `svc/vllm-jozu:8000` · SGLang `svc/sglang-jozu:8001`.

### Mode B — Jozu RIC (turnkey image)

A **Runtime Inference Container** is auto-generated by Jozu Hub: one signed OCI
image with CUDA + vLLM + the model weights baked in. No init container, no unpack.

```
image: jozu.ml/jonathangamer202002/qwen3-4b-instruct/vllm:latest   # ~17.6 GB, anonymous pull
```

Its entrypoint runs `vllm serve "${MODEL_PATH:-/}" --served-model-name "${SERVING_NAME[@]}" "$@"`.
Because `MODEL_PATH` isn't set in the image **and** this ModelKit uses the
subdir layout, `vllm-ric-jozu.yaml` (a) sets `MODEL_PATH=/qwen3-4b-instruct` and
(b) symlinks the shards up beside the index before handing off to the stock entrypoint:

```yaml
env:
  - { name: MODEL_PATH, value: /qwen3-4b-instruct }
command: ["/bin/bash", "-lc"]
args:
  - cd /qwen3-4b-instruct && for f in model/*.safetensors; do ln -sf "$f" .; done &&
    exec /usr/local/bin/entrypoint.sh --gpu-memory-utilization=0.85 --max-model-len=8192
```

Served model name: **`model`** (and `jonathangamer202002/qwen3-4b-instruct`) · `svc/vllm-ric-jozu:8000`.

---

## HAMi GPU virtualization & caps

HAMi enforces caps at the CUDA/NVML layer via the scheduler + device plugin
(injecting a specific vGPU UUID and the `libvgpu` preload) — **independent of what
is inside the image**. This is why it works even for the turnkey RIC, whose image
ships `NVIDIA_VISIBLE_DEVICES=all`; HAMi overrides it with a single vGPU.

Inside any HAMi pod you can see the injected limits:

```bash
kubectl exec deploy/vllm-ric-jozu -- sh -c 'env | grep -E "CUDA_DEVICE_(MEMORY|SM)|NVIDIA_VISIBLE"'
# CUDA_DEVICE_MEMORY_LIMIT_0=48935m      <- memory cap
# CUDA_DEVICE_SM_LIMIT=60                <- compute cap (%)
# NVIDIA_VISIBLE_DEVICES=GPU-<uuid>      <- a single vGPU, not "all"
```

### Capping a workload to 60% of the GPU

`kitops/charts/vllm-ric-jozu.yaml` caps the RIC to **60% of the H100**:

```yaml
resources:
  limits:
    nvidia.com/gpu: "1"
    nvidia.com/gpucores: "60"            # 60% of the SMs (compute)
    nvidia.com/gpumem-percentage: "60"   # 60% of 80 GB  (~48 GB)
```

**Evidence it is enforced** (verified live):

```bash
# Inside the pod: 60% of 81559 MiB
kubectl exec deploy/vllm-ric-jozu -- nvidia-smi --query-gpu=memory.total,memory.used --format=csv
#  -> NVIDIA H100 80GB HBM3, 48935 MiB, 42515 MiB

# On the host: full 80 GB, the RIC sits inside its 48.9 GB slice (~40% GPU free)
nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv
#  -> <pid>, 43244 MiB, VLLM::EngineCore
```

> Knobs are independent: e.g. 60% compute with a fixed 30 GB =
> `gpucores: "60"` + `gpumem: "30000"`. The remaining capacity stays free for
> other pods — that is the GPU-sharing story.

---

## Build & deploy

> Docker on this host runs via the `docker` group: prefix with `sg docker -c '…'` if needed.

### 1. Build the Mode A images and load them into kind

```bash
cd kitops && ./build.sh          # builds + `kind load`s:
#   hami-kitunpacker:latest   (alpine + kit CLI + unpack/flatten)
#   hami-vllm-jozu:latest      (vllm/vllm-openai + custom serve cmd)
#   hami-sglang-jozu:latest    (lmsysorg/sglang + custom serve cmd)
# or:  make kitops-build
```

### 2. Deploy

```bash
# Mode A — ModelKit + unpack (vLLM + SGLang). Frees GPU from the old HF qwen/sglang.
make kitops-deploy
# Watch the ModelKit get pulled from Jozu Hub by the init container:
kubectl logs -l app=vllm-jozu -c kitops-init -f

# Mode B — the turnkey RIC (scales Mode-A vLLM down to make room)
make kitops-ric
```

### Makefile targets

| Target | Action |
| --- | --- |
| `make kitops-build` | build + `kind load` the Mode-A images |
| `make kitops-deploy` | deploy `vllm-jozu` + `sglang-jozu` (Mode A) |
| `make kitops-ric` | deploy `vllm-ric-jozu` (Mode B) |
| `make kitops-demo` | run the guided demo (`DEPLOY=1` applies manifests) |
| `make kitops-destroy` | delete all KitOps/Jozu deployments |

---

## Running the guided demo

```bash
# interactive (pauses between acts)
./scripts/hami_kitops_demo.sh

# straight through, applying manifests as it goes
AUTO=1 DEPLOY=1 ./scripts/hami_kitops_demo.sh
```

The script auto-detects which vLLM is up (RIC vs unpack) and walks through:
1. Model provenance on Jozu Hub (`kit inspect --remote`, no login)
2. The custom serve command + the kitunpacker / RIC images
3. Deploy on HAMi — the init container pulling from Jozu Hub
4. HAMi virtualization — engines co-resident on one H100
5. Live inference on both vLLM and SGLang
6. Pod-capped `nvidia-smi` + per-vGPU utilization (`scripts/vgpu_util.py`)

Useful env vars: `AUTO=1`, `DEPLOY=1`, `VLLM_MODE=ric|unpack|auto`,
`FREE_GPU=1`, `NS=default`.

---

## Inference examples

```bash
# Mode B — RIC (served model name is "model")
kubectl port-forward svc/vllm-ric-jozu 8000:8000 &
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"model","messages":[{"role":"user","content":"What is GPU virtualization?"}],"max_tokens":80}'

# Mode A — vLLM (served model name is "qwen3-4b-instruct")
kubectl port-forward svc/vllm-jozu 8000:8000 &
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-4b-instruct","messages":[{"role":"user","content":"hi"}]}'

# Mode A — SGLang (svc port 8001 -> container 30000)
kubectl port-forward svc/sglang-jozu 8001:8001 &
curl -s localhost:8001/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-4b-instruct","messages":[{"role":"user","content":"hi"}]}'
```

---

## File map

```
kubecon-demo/
├── demo.md                          # this document
├── Makefile                         # kitops-build / -deploy / -ric / -demo / -destroy
├── kind-gpu.yaml                    # kind cluster with GPU passthrough
├── kitops/                          # the KitOps + Jozu Hub pipeline
│   ├── README.md
│   ├── Kitfile                      # reference manifest to (re)pack + push your own ModelKit
│   ├── build.sh                     # build 3 images + kind load
│   ├── docker/
│   │   ├── kitunpacker/             # init container: pull + unpack + flatten ModelKit
│   │   │   ├── Dockerfile
│   │   │   └── unpack.sh
│   │   ├── vllm/                    # custom vLLM serve command
│   │   │   ├── Dockerfile
│   │   │   └── serve.sh
│   │   └── sglang/                  # custom SGLang serve command
│   │       ├── Dockerfile
│   │       └── serve.sh
│   └── charts/
│       ├── vllm-jozu.yaml           # Mode A: vLLM (init + custom image)
│       ├── sglang-jozu.yaml         # Mode A: SGLang (init + custom image)
│       └── vllm-ric-jozu.yaml       # Mode B: Jozu RIC, capped to 60%
├── charts/                          # earlier HF-based + HAMi demo workloads
│   ├── qwen/qwen.yaml               # vLLM Qwen3-1.7B (HF runtime download)
│   ├── sglang/sglang.yaml           # SGLang Qwen3-1.7B (HF runtime download)
│   ├── fanout/fanout.yaml           # many small pods sharing one H100
│   ├── monitoring/prometheus.yaml   # Prometheus scraping HAMi + DCGM
│   └── gpu-smoke.yaml               # quick HAMi memory-cap smoke test
└── scripts/
    ├── hami_kitops_demo.sh          # guided KitOps/Jozu demo (Mode A + B aware)
    ├── hami_sglang_demo.sh          # guided HAMi + SGLang demo
    └── vgpu_util.py                 # per-vGPU utilization from HAMi metrics
```

---

## Troubleshooting

| Symptom | Cause / fix |
| --- | --- |
| RIC crash-loops: `Invalid … directory specified: '/'` | `MODEL_PATH` unset + subdir layout. Use the `MODEL_PATH` + symlink override in `vllm-ric-jozu.yaml`. |
| New RIC pod stuck `Pending` after edit | Old pod still holds the single GPU. `kubectl scale deploy/vllm-ric-jozu --replicas=0`, wait, then `--replicas=1`. |
| vLLM/SGLang OOM at startup | Lower `--gpu-memory-utilization` or raise the HAMi `gpumem` cap. |
| init container re-pulls 8 GB every start | Confirm the shared `hostPath` cache; `unpack.sh` skips when `/models/qwen3` already exists. |
| In-pod `nvidia-smi` shows full 80 GB | Pod isn't using `schedulerName: hami-scheduler`, or no `nvidia.com/gpumem*` limit. |
| `kit inspect`/pull fails | It uses the standard registry token flow; public repos need no login, private repos need `kit login` (env `REGISTRY_URL/USERNAME/PASSWORD` on `kitunpacker`). |

---

### Two Jozu modes, one takeaway

- **ModelKit + unpack** keeps the *model (data)* separate from the *runtime*, lets you
  bring your own serve command, and reuse one cached model across vLLM **and** SGLang.
- **RIC** is the fastest path: model + server as a single signed artifact.

Either way, **HAMi virtualizes the H100** so the workload runs inside a capped
slice (e.g. 60%), leaving the rest of the GPU free for other tenants.
