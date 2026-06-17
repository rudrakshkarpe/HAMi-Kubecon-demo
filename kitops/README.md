# KitOps + Jozu Hub model pipeline (HAMi + vLLM + SGLang)

This pipeline replaces the runtime **Hugging Face download** in the original demo
with a **KitOps ModelKit pulled from the [Jozu Hub](https://jozu.ml) registry**.
The model is delivered into the Pod by a KitOps `initContainer`, and both vLLM
and SGLang serve it from a **local directory** using a **custom command baked
into their Dockerfiles** — no `vllm serve <hf-repo>` and no HF download.

Adapted from the Azure `aks-byo-models-kaito` lab (KitOps + initContainer
pattern), but targeting our `kind` + HAMi cluster and public Jozu Hub pulls.

## The model

Public, login-free ModelKit on Jozu Hub (safetensors, `Qwen3ForCausalLM`):

```
jozu.ml/jonathangamer202002/qwen3-4b-instruct:latest
```

```bash
kit inspect --remote jozu.ml/jonathangamer202002/qwen3-4b-instruct:latest
```

## Architecture

```
            Jozu Hub (OCI registry)
                    │  kit unpack --filter model   (public, no login)
                    ▼
┌───────────────────────────────────────────────────────────┐
│ Pod (scheduled by hami-scheduler on the H100)              │
│                                                            │
│  initContainer: hami-kitunpacker                           │
│    • kit unpack ModelKit -> /models                        │
│    • flatten to HF layout  -> /models/qwen3                │
│                    │ shared volume                         │
│                    ▼                                       │
│  container: hami-vllm-jozu / hami-sglang-jozu              │
│    • custom entrypoint:  serve /models/qwen3               │
│    • HAMi limits: gpumem=30000, gpucores=30                │
└───────────────────────────────────────────────────────────┘
```

The `.safetensors` shards live in a `model/` subdir inside the ModelKit while
`config.json` / `*.index.json` / tokenizer sit one level up; the unpacker
**flattens** them into one directory that vLLM/transformers can load.

## Files

| Path | Purpose |
| --- | --- |
| `docker/kitunpacker/` | Alpine + `kit` CLI init image; pulls + unpacks + flattens the ModelKit |
| `docker/vllm/` | `vllm/vllm-openai` + custom `serve.sh` (serves the local model) |
| `docker/sglang/` | `lmsysorg/sglang` + custom `serve.sh` (serves the local model) |
| `charts/vllm-jozu.yaml` | vLLM Deployment + Service (init container + HAMi limits) |
| `charts/sglang-jozu.yaml` | SGLang Deployment + Service |
| `Kitfile` | Reference manifest to (re)pack + push your own ModelKit |
| `build.sh` | Build the 3 images and `kind load` them |

## Run it

```bash
# 1) build images + load into kind  (host may need: sg docker -c '...')
cd kitops && ./build.sh

# 2) free GPU memory used by the old HF deployments, then deploy
kubectl scale deploy/qwen deploy/sglang --replicas=0
kubectl apply -f charts/vllm-jozu.yaml -f charts/sglang-jozu.yaml

# 3) watch the ModelKit get pulled from Jozu Hub by the init container
kubectl logs -l app=vllm-jozu -c kitops-init -f

# 4) once Running, test inference (served-model-name = qwen3-4b-instruct)
kubectl port-forward svc/vllm-jozu 18000:8000 &
curl -s localhost:18000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-4b-instruct","messages":[{"role":"user","content":"hi"}]}'
```

Or run the guided demo:

```bash
AUTO=1 DEPLOY=1 ./scripts/hami_kitops_demo.sh
```

## Two ways to consume the model from Jozu Hub

Jozu Hub exposes the same model in two forms. This repo supports both:

| Mode | Image / ref | How the model arrives | Custom command? |
| --- | --- | --- | --- |
| **ModelKit + unpack** (`vllm-jozu.yaml`, `sglang-jozu.yaml`) | `…/qwen3-4b-instruct:latest` (data artifact) | `kitunpacker` initContainer pulls + unpacks safetensors to a shared volume | Yes — baked into our `serve.sh`/Dockerfile |
| **RIC** (`vllm-ric-jozu.yaml`) | `…/qwen3-4b-instruct/vllm:latest` (runnable image) | Model weights are **baked into the image**; the RIC entrypoint starts vLLM on :8000 | No — turnkey, just pull + run |

Both are **public / anonymous-pullable**. The RIC is the simplest path (one
signed OCI artifact = model + server); the ModelKit+unpack path keeps the model
(data) separate from the runtime and lets you bring your own serve command and
reuse one model across vLLM **and** SGLang.

```bash
# RIC: pull straight into the node, then deploy on HAMi
kubectl apply -f charts/vllm-ric-jozu.yaml     # image: jozu.ml/.../vllm:latest
# or:  make kitops-ric
```

## Using a different model / a private registry

* **Different ModelKit:** override `MODELKIT_REF` (and `MODEL_SUBDIR`) env on the
  `kitops-init` container in the charts.
* **Private registry (e.g. Jozu Hub private repo / ACR):** add
  `REGISTRY_URL`, `USERNAME`, `PASSWORD` env (e.g. from a Secret) to
  `kitops-init`; `unpack.sh` runs `kit login` before pulling.
* **Push your own:** put a HF model dir at `./qwen3` next to the `Kitfile`, then
  `kit pack . -t jozu.ml/<org>/<repo>:latest && kit push jozu.ml/<org>/<repo>:latest`.
