#!/usr/bin/env bash
#
# HAMi + KitOps + Jozu Hub end-to-end demo
# ----------------------------------------
# Same HAMi-virtualized H100, but the model now comes from a KitOps ModelKit on
# the Jozu Hub registry instead of a runtime Hugging Face download. Story:
#   1. Model provenance: the ModelKit on Jozu Hub (safetensors, no login, no HF)
#   2. The custom serve command baked into the Dockerfiles + the kitunpacker init
#   3. Deploy on HAMi: watch the initContainer pull + unpack from Jozu Hub
#   4. HAMi virtualization: both engines co-resident on one H100
#   5. Live inference -- same KitOps model served by vLLM AND SGLang
#   6. Pod-capped nvidia-smi + per-vGPU utilization
#
# Usage:
#   ./scripts/hami_kitops_demo.sh           # interactive (pauses between acts)
#   AUTO=1 ./scripts/hami_kitops_demo.sh    # run straight through
#   DEPLOY=1 ./scripts/hami_kitops_demo.sh  # (re)apply manifests during Act 3
#
# Env overrides:
#   NS=default  MODELKIT_REF=jozu.ml/jonathangamer202002/qwen3-4b-instruct:latest
#   SERVED_NAME=qwen3-4b-instruct  NODE=hami-demo-control-plane
#   VLLM_LPORT=18000  SGLANG_LPORT=18001  FREE_GPU=1 (scale down HF qwen/sglang)
set -uo pipefail

# ----------------------------- config -----------------------------
NS="${NS:-default}"
MODELKIT_REF="${MODELKIT_REF:-jozu.ml/jonathangamer202002/qwen3-4b-instruct:latest}"
SERVED_NAME="${SERVED_NAME:-qwen3-4b-instruct}"
NODE="${NODE:-hami-demo-control-plane}"
VLLM_LPORT="${VLLM_LPORT:-18000}"
SGLANG_LPORT="${SGLANG_LPORT:-18001}"
# vLLM source: "ric" = turnkey Jozu RIC (model baked in), "unpack" = ModelKit
# pulled+unpacked by the kitunpacker init container. "auto" picks whichever is up.
VLLM_MODE="${VLLM_MODE:-auto}"
AUTO="${AUTO:-}"
DEPLOY="${DEPLOY:-}"
FREE_GPU="${FREE_GPU:-1}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CHARTS="$REPO_DIR/kitops/charts"

# ----------------------------- colors -----------------------------
if [ -t 1 ]; then
  BOLD=$(tput bold); DIM=$(tput dim); RESET=$(tput sgr0)
  RED=$(tput setaf 1); GREEN=$(tput setaf 2); YELLOW=$(tput setaf 3)
  BLUE=$(tput setaf 4); MAGENTA=$(tput setaf 5); CYAN=$(tput setaf 6)
else
  BOLD=""; DIM=""; RESET=""; RED=""; GREEN=""; YELLOW=""; BLUE=""; MAGENTA=""; CYAN=""
fi

VLLM_PF_PID=""; SGLANG_PF_PID=""
cleanup() {
  [ -n "$VLLM_PF_PID" ]   && kill "$VLLM_PF_PID"   >/dev/null 2>&1
  [ -n "$SGLANG_PF_PID" ] && kill "$SGLANG_PF_PID" >/dev/null 2>&1
}
trap cleanup EXIT INT TERM

# ----------------------------- helpers -----------------------------
hr()   { printf "${DIM}%s${RESET}\n" "------------------------------------------------------------------------"; }
act()  { echo; printf "${BOLD}${MAGENTA}╔══════════════════════════════════════════════════════════════════════╗${RESET}\n";
         printf "${BOLD}${MAGENTA}║  %-66s║${RESET}\n" "$1";
         printf "${BOLD}${MAGENTA}╚══════════════════════════════════════════════════════════════════════╝${RESET}\n"; }
note() { printf "${CYAN}▸ %s${RESET}\n" "$*"; }
ok()   { printf "${GREEN}✓ %s${RESET}\n" "$*"; }
warn() { printf "${YELLOW}! %s${RESET}\n" "$*"; }
pause(){ [ -n "$AUTO" ] && { sleep "${AUTO_SLEEP:-1}"; return; }
         printf "\n${DIM}   ── press ${RESET}${BOLD}ENTER${RESET}${DIM} to continue ──${RESET}"; read -r _; }
run()  { printf "\n${GREEN}\$ %s${RESET}\n" "$*"; eval "$@"; }
require(){ command -v "$1" >/dev/null 2>&1 || { echo "${RED}Missing required tool: $1${RESET}"; exit 1; }; }

# =====================================================================
clear 2>/dev/null || true
printf "${BOLD}${BLUE}"
cat <<'BANNER'
  _   _    _    __  __ _    _  _____ _____ ___  ____  ____
 | | | |  / \  |  \/  (_)  | |/ /_ _|_   _/ _ \|  _ \/ ___|
 | |_| | / _ \ | |\/| | |  | ' / | |  | || | | | |_) \___ \
 |  _  |/ ___ \| |  | | |  | . \ | |  | || |_| |  __/ ___) |
 |_| |_/_/   \_\_|  |_|_|  |_|\_\___| |_| \___/|_|   |____/
   ModelKits from Jozu Hub  ->  vLLM + SGLang  on a HAMi H100
BANNER
printf "${RESET}\n"
note "Registry:  Jozu Hub  (${MODELKIT_REF})"
note "Model:     ${SERVED_NAME}  (safetensors, pulled by a KitOps initContainer)"
note "Engines:   vLLM (:8000) and SGLang (:30000), same local model, no HF download"
note "Mode:      $([ -n "$AUTO" ] && echo 'AUTO (no pauses)' || echo 'interactive')"
pause

require kubectl; require nvidia-smi; require curl
KIT_OK=1; command -v kit >/dev/null 2>&1 || KIT_OK=""

# Resolve which vLLM is serving: the turnkey Jozu RIC or the unpack-flow image.
deploy_ready() { [ "$(kubectl -n "$NS" get deploy "$1" -o jsonpath='{.status.readyReplicas}' 2>/dev/null)" = "1" ]; }
VLLM_SVC="vllm-jozu"; VLLM_MODEL="$SERVED_NAME"; VLLM_KIND="ModelKit unpack + custom command"
if { [ "$VLLM_MODE" = "ric" ] || [ "$VLLM_MODE" = "auto" ]; } && deploy_ready vllm-ric-jozu; then
  VLLM_SVC="vllm-ric-jozu"; VLLM_MODEL="model"; VLLM_KIND="Jozu RIC (model baked into the image)"
fi

# ---------------------------------------------------------------------
act "ACT 1 -- Model provenance: a ModelKit on Jozu Hub (not Hugging Face)"
note "The model is packaged as an OCI ModelKit and lives in the Jozu Hub registry."
note "It is PULLED from there at deploy time -- there is no Hugging Face download."
if [ -n "$KIT_OK" ]; then
  run "kit inspect --remote $MODELKIT_REF | sed -n '1,40p'"
  note "Note the model layer mediaType 'modelkit.model.v1.tar' + safetensors parts,"
  note "license Apache-2.0, and that 'kit inspect --remote' worked with NO login."
else
  warn "kit CLI not found locally; skipping live inspect (the cluster init container still uses it)."
fi
pause

# ---------------------------------------------------------------------
act "ACT 2 -- The custom command (baked in the Dockerfiles)"
note "Default HF flow:   vllm serve Qwen/Qwen3-1.7B          (downloads from HF)"
note "Our KitOps flow:   vllm serve /models/qwen3            (local, from the ModelKit)"
hr
note "initContainer image -- pulls + unpacks the ModelKit from Jozu Hub:"
run "sed -n '1,20p' $REPO_DIR/kitops/docker/kitunpacker/Dockerfile"
note "Custom vLLM serve command (entrypoint baked into the image):"
run "grep -A6 'exec vllm serve' $REPO_DIR/kitops/docker/vllm/serve.sh"
note "Custom SGLang serve command:"
run "grep -A8 'exec python3' $REPO_DIR/kitops/docker/sglang/serve.sh"
hr
note "Alternative -- the turnkey Jozu RIC (model + vLLM baked into ONE image):"
note "  image: jozu.ml/jonathangamer202002/qwen3-4b-instruct/vllm:latest  (anonymous pull)"
note "  no init container, no unpack -- its entrypoint serves vllm on :8000."
note "Active vLLM path for this run: ${VLLM_KIND} (svc/${VLLM_SVC})"
pause

# ---------------------------------------------------------------------
act "ACT 3 -- Deploy on HAMi: initContainer pulls + unpacks from Jozu Hub"
if [ -n "$FREE_GPU" ]; then
  note "Freeing GPU memory: scaling down the old HF-based qwen/sglang (if present)."
  kubectl -n "$NS" scale deploy/qwen   --replicas=0 >/dev/null 2>&1 && ok "scaled qwen -> 0"   || true
  kubectl -n "$NS" scale deploy/sglang --replicas=0 >/dev/null 2>&1 && ok "scaled sglang -> 0" || true
fi
if [ -n "$DEPLOY" ]; then
  run "kubectl -n $NS apply -f $CHARTS/vllm-jozu.yaml"
  run "kubectl -n $NS apply -f $CHARTS/sglang-jozu.yaml"
else
  note "(set DEPLOY=1 to apply manifests here; assuming they are already applied)"
fi
hr
note "Watching the kitops-init container pull the ModelKit from Jozu Hub..."
VPOD="$(kubectl -n "$NS" get pod -l app=vllm-jozu -o name 2>/dev/null | head -1)"
if [ -n "$VPOD" ]; then
  run "kubectl -n $NS logs $VPOD -c kitops-init --tail=20 2>/dev/null || echo '(init not started yet)'"
fi
note "Unpacked Hugging Face-layout model on the node volume:"
run "kubectl exec -n $NS ${VPOD:-deploy/vllm-jozu} -c vllm-jozu -- ls -la /models/qwen3 2>/dev/null | head -15 || echo '(pod not ready yet)'"
pause

# ---------------------------------------------------------------------
act "ACT 4 -- HAMi virtualization: both engines on ONE physical H100"
run "kubectl -n $NS get pods -l pipeline=kitops-jozu -o wide"
note "HAMi advertises the single H100 as multiple schedulable vGPUs:"
run "kubectl get node $NODE -o jsonpath='{.status.capacity.nvidia\\.com/gpu}{\"\\n\"}'"
note "Host nvidia-smi -- the real 80GB H100 with BOTH engines co-resident:"
run "nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv"
run "nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv | head -12"
pause

# ---------------------------------------------------------------------
act "ACT 5 -- Live inference: same Jozu Hub model, two engines"
note "vLLM OpenAI API (:8000) <- ${VLLM_KIND}  [svc/${VLLM_SVC}, model='${VLLM_MODEL}']"
kubectl -n "$NS" port-forward "svc/${VLLM_SVC}" "${VLLM_LPORT}:8000" >/dev/null 2>&1 &
VLLM_PF_PID=$!; sleep 4
run "curl -s http://localhost:${VLLM_LPORT}/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{\"model\":\"${VLLM_MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"In one sentence, what is GPU virtualization?\"}],\"max_tokens\":80}' \
  | python3 -m json.tool 2>/dev/null | grep -A2 '\"content\"' || echo '(vLLM not ready)'"
hr
note "SGLang OpenAI API (:30000 -> svc :8001) <- the SAME model"
kubectl -n "$NS" port-forward svc/sglang-jozu "${SGLANG_LPORT}:8001" >/dev/null 2>&1 &
SGLANG_PF_PID=$!; sleep 4
run "curl -s http://localhost:${SGLANG_LPORT}/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{\"model\":\"${SERVED_NAME}\",\"messages\":[{\"role\":\"user\",\"content\":\"Name one benefit of packaging models as OCI artifacts.\"}],\"max_tokens\":80}' \
  | python3 -m json.tool 2>/dev/null | grep -A2 '\"content\"' || echo '(SGLang not ready)'"
pause

# ---------------------------------------------------------------------
act "ACT 6 -- Pod-capped nvidia-smi + per-vGPU utilization"
note "Inside the vLLM pod (svc/${VLLM_SVC}), HAMi caps the visible GPU memory (not the full 80GB):"
run "kubectl exec -n $NS deploy/${VLLM_SVC} -- nvidia-smi --query-gpu=memory.total,memory.used --format=csv 2>/dev/null || echo '(pod not ready)'"
if [ -f "$SCRIPT_DIR/vgpu_util.py" ]; then
  note "Per-vGPU utilization attributed by HAMi (what nvidia-smi alone cannot show):"
  run "python3 $SCRIPT_DIR/vgpu_util.py 2>/dev/null || echo '(metrics endpoint unavailable)'"
fi
pause

# ---------------------------------------------------------------------
act "RECAP"
ok "Model packaged as an OCI ModelKit, stored on Jozu Hub (no Hugging Face at runtime)"
ok "kitunpacker initContainer pulls + unpacks it into a shared volume"
ok "vLLM and SGLang serve the LOCAL model via a custom command baked in the Dockerfile"
ok "HAMi virtualizes one H100 so both engines share it with capped memory/cores"
echo
note "Re-run live:  AUTO=1 DEPLOY=1 ./scripts/hami_kitops_demo.sh"
echo
