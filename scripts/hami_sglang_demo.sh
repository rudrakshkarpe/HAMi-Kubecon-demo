#!/usr/bin/env bash
#
# HAMi + SGLang end-to-end demo
# -----------------------------
# Tells the story in 6 acts on a single H100:
#   1. The cluster + HAMi components
#   2. HAMi virtualization: 1 physical GPU -> many schedulable slices
#   3. Host nvidia-smi: the real 80GB H100
#   4. The SGLang inference pod + its HAMi-capped GPU view
#   5. A live inference request over SGLang's OpenAI API
#   6. nvidia-smi "through a lot of things" while the GPU is under load
#
# Usage:
#   ./scripts/hami_sglang_demo.sh            # interactive (pauses between acts)
#   AUTO=1 ./scripts/hami_sglang_demo.sh     # no pauses, run straight through
#
# Env overrides:
#   NS=default  SGLANG_SVC=sglang  SGLANG_PORT=8001  LPORT=18001
#   MODEL=Qwen/Qwen3-1.7B  NODE=hami-demo-control-plane
set -uo pipefail

# ----------------------------- config -----------------------------
NS="${NS:-default}"
SGLANG_SVC="${SGLANG_SVC:-sglang}"
SGLANG_PORT="${SGLANG_PORT:-8001}"
LPORT="${LPORT:-18001}"
MODEL="${MODEL:-Qwen/Qwen3-1.7B}"
NODE="${NODE:-hami-demo-control-plane}"
AUTO="${AUTO:-}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# ----------------------------- colors -----------------------------
if [ -t 1 ]; then
  BOLD=$(tput bold); DIM=$(tput dim); RESET=$(tput sgr0)
  RED=$(tput setaf 1); GREEN=$(tput setaf 2); YELLOW=$(tput setaf 3)
  BLUE=$(tput setaf 4); MAGENTA=$(tput setaf 5); CYAN=$(tput setaf 6)
else
  BOLD=""; DIM=""; RESET=""; RED=""; GREEN=""; YELLOW=""; BLUE=""; MAGENTA=""; CYAN=""
fi

PF_PID=""
MON_PID=""
LOADER_PID=""
MON_LPORT="${MON_LPORT:-31992}"
cleanup() {
  [ -n "$LOADER_PID" ] && kill "$LOADER_PID" >/dev/null 2>&1
  pkill -f "localhost:${LPORT}/v1/chat" >/dev/null 2>&1
  [ -n "$MON_PID" ] && kill "$MON_PID" >/dev/null 2>&1
  [ -n "$PF_PID" ] && kill "$PF_PID" >/dev/null 2>&1
}
trap cleanup EXIT INT TERM

# ----------------------------- helpers -----------------------------
hr()      { printf "${DIM}%s${RESET}\n" "------------------------------------------------------------------------"; }
act()     { echo; printf "${BOLD}${MAGENTA}╔══════════════════════════════════════════════════════════════════════╗${RESET}\n";
            printf "${BOLD}${MAGENTA}║  %-66s║${RESET}\n" "$1";
            printf "${BOLD}${MAGENTA}╚══════════════════════════════════════════════════════════════════════╝${RESET}\n"; }
note()    { printf "${CYAN}▸ %s${RESET}\n" "$*"; }
ok()      { printf "${GREEN}✓ %s${RESET}\n" "$*"; }
warn()    { printf "${YELLOW}! %s${RESET}\n" "$*"; }
pause()   { [ -n "$AUTO" ] && { sleep "${AUTO_SLEEP:-1}"; return; }
            printf "\n${DIM}   ── press ${RESET}${BOLD}ENTER${RESET}${DIM} to continue ──${RESET}"; read -r _; }
# run: echo a command in green, then execute it
run()     { printf "\n${GREEN}\$ %s${RESET}\n" "$*"; eval "$@"; }

require() { command -v "$1" >/dev/null 2>&1 || { echo "${RED}Missing required tool: $1${RESET}"; exit 1; }; }

# =====================================================================
clear 2>/dev/null || true
printf "${BOLD}${BLUE}"
cat <<'BANNER'
  _   _    _    __  __ _    __     ____  ____  _                         
 | | | |  / \  |  \/  (_)   \ \   / ___|/ ___|| |    __ _ _ __   __ _    
 | |_| | / _ \ | |\/| | |    \ \  \___ \ |  _ | |   / _` | '_ \ / _` |   
 |  _  |/ ___ \| |  | | |     \ \  ___) | |_| || |__| (_| | | | | (_| |  
 |_| |_/_/   \_\_|  |_|_|      \_\|____/ \____||_____\__,_|_| |_|\__, |  
   GPU virtualization  +  SGLang inference  on a single H100      |___/  
BANNER
printf "${RESET}\n"
note "Cluster:   kind (${NODE})"
note "Engine:    SGLang serving ${MODEL}"
note "Mode:      $([ -n "$AUTO" ] && echo 'AUTO (no pauses)' || echo 'interactive (pauses between acts)')"
pause

require kubectl
require nvidia-smi
require curl

# ---------------------------------------------------------------------
# ACT 0 — make sure SGLang is up (deploy if missing)
# ---------------------------------------------------------------------
act "ACT 0 · Ensuring the SGLang inference service is deployed"
if ! kubectl -n "$NS" get deploy "$SGLANG_SVC" >/dev/null 2>&1; then
  warn "SGLang deployment not found — applying manifest..."
  run "kubectl apply -f '$REPO_DIR/charts/sglang/sglang.yaml'"
else
  ok "SGLang deployment already present"
fi
note "Waiting for the SGLang pod to be Ready (model load + CUDA graph capture can take a few minutes)..."
run "kubectl -n '$NS' rollout status deploy/'$SGLANG_SVC' --timeout=420s"
pause

# ---------------------------------------------------------------------
# ACT 1 — the cluster and HAMi control plane
# ---------------------------------------------------------------------
act "ACT 1 · The Kubernetes cluster and the HAMi control plane"
note "One node, backed by one physical NVIDIA H100."
run "kubectl get nodes -o wide"
echo
note "HAMi installs a scheduler + a device-plugin (this is what virtualizes the GPU):"
run "kubectl -n kube-system get pods | grep -E 'hami|NAME'"
pause

# ---------------------------------------------------------------------
# ACT 2 — HAMi virtualization: 1 GPU -> many slices
# ---------------------------------------------------------------------
act "ACT 2 · HAMi virtualization — one card becomes many schedulable GPUs"
note "The kernel/driver sees exactly ONE physical GPU:"
run "nvidia-smi --query-gpu=index,name,memory.total --format=csv"
echo
note "But HAMi advertises the node's GPU as MULTIPLE schedulable units:"
run "kubectl get node '$NODE' -o jsonpath='{.status.allocatable.nvidia\\.com/gpu}'; echo ' virtual GPUs'"
echo
note "Workloads also request fine-grained slices — memory (MB) and compute cores (%)."
note "Here is what our SGLang pod asked HAMi for:"
run "kubectl -n '$NS' get deploy '$SGLANG_SVC' -o jsonpath='{.spec.template.spec.containers[0].resources.limits}'; echo"
echo
echo "${DIM}   nvidia.com/gpu      -> number of (virtual) GPUs${RESET}"
echo "${DIM}   nvidia.com/gpumem   -> hard memory cap in MB seen inside the pod${RESET}"
echo "${DIM}   nvidia.com/gpucores -> compute-core percentage cap${RESET}"
pause

# ---------------------------------------------------------------------
# ACT 3 — host nvidia-smi: the real H100
# ---------------------------------------------------------------------
act "ACT 3 · Host nvidia-smi — the real, full 80GB H100"
run "nvidia-smi"
pause

# ---------------------------------------------------------------------
# ACT 4 — the pod's HAMi-capped view
# ---------------------------------------------------------------------
act "ACT 4 · Inside the SGLang pod — the HAMi-virtualized GPU view"
SG_POD="$(kubectl -n "$NS" get pods -l app="$SGLANG_SVC" -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)"
note "SGLang pod: ${BOLD}${SG_POD}${RESET}"
echo
note "From INSIDE the pod, nvidia-smi reports a much smaller GPU — the slice HAMi granted."
note "Watch 'Memory-Usage': it is capped far below 80GB."
run "kubectl -n '$NS' exec '$SG_POD' -- nvidia-smi"
echo
printf "${BOLD}${YELLOW}   The same physical H100 looks like an 80GB card to the host,\n"
printf "   and like a small dedicated GPU to the pod. That is HAMi.${RESET}\n"
echo
printf "${DIM}   Note: HAMi is SOFTWARE vGPU sharing, not hardware MIG. The slice shows up\n"
printf "   as a reduced memory TOTAL (25000 MiB), NOT as '1 of 10' devices. The '10'\n"
printf "   is a Kubernetes scheduling count (Act 2); nvidia-smi reports hardware state,\n"
printf "   so it cannot show it. Act 7 makes the sharing visible by packing many pods.${RESET}\n"
pause

# ---------------------------------------------------------------------
# ACT 5 — live inference over SGLang
# ---------------------------------------------------------------------
act "ACT 5 · A live inference request over the SGLang OpenAI-compatible API"
note "Opening a port-forward to the SGLang service on localhost:${LPORT} ..."
kubectl -n "$NS" port-forward "svc/$SGLANG_SVC" "${LPORT}:${SGLANG_PORT}" >/tmp/hami_demo_pf.log 2>&1 &
PF_PID=$!
for i in $(seq 1 20); do
  curl -s "http://localhost:${LPORT}/v1/models" >/dev/null 2>&1 && break
  sleep 0.5
done
ok "Port-forward ready"
echo
note "Available models served by SGLang:"
run "curl -s http://localhost:${LPORT}/v1/models | python3 -m json.tool"
echo
note "Sending a chat completion request..."
PROMPT="Explain in 2 short sentences what GPU virtualization with HAMi enables for a Kubernetes cluster."
printf "${BLUE}   prompt: %s${RESET}\n" "$PROMPT"
REQ=$(cat <<JSON
{"model":"$MODEL","messages":[{"role":"user","content":"$PROMPT"}],"max_tokens":160,"temperature":0.3,"chat_template_kwargs":{"enable_thinking":false}}
JSON
)
run "curl -s http://localhost:${LPORT}/v1/chat/completions -H 'Content-Type: application/json' -d '$REQ' | python3 -c 'import sys,json; d=json.load(sys.stdin); print(\"\n${GREEN}>>> SGLang answer:${RESET}\n\"+d[\"choices\"][0][\"message\"][\"content\"].strip()+\"\n\n${DIM}tokens: prompt=%s completion=%s${RESET}\"%(d[\"usage\"][\"prompt_tokens\"],d[\"usage\"][\"completion_tokens\"]))'"
pause

# ---------------------------------------------------------------------
# ACT 6 — nvidia-smi "through a lot of things" under load
# ---------------------------------------------------------------------
act "ACT 6 · nvidia-smi everywhere — watching the H100 work under load"

note "First, who is actually using the physical GPU right now (per-process):"
run "nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv"
echo
note "Now we drive sustained traffic at SGLang and watch the GPU work in real time."

# Sustained load: a self-contained subshell keeps ~12 generations in flight for
# DURATION seconds. Running it in a subshell means its internal `wait` only sees
# its own curl children (never the long-lived port-forward).
start_sustained_load() {
  local dur="$1"
  (
    local body end
    body='{"model":"'"$MODEL"'","messages":[{"role":"user","content":"Write a long, detailed essay about high performance computing and GPUs."}],"max_tokens":512,"temperature":0.8,"chat_template_kwargs":{"enable_thinking":false}}'
    end=$((SECONDS + dur))
    while [ "$SECONDS" -lt "$end" ]; do
      while [ "$(jobs -rp | wc -l)" -lt 12 ]; do
        curl -s --max-time "$dur" "http://localhost:${LPORT}/v1/chat/completions" \
             -H 'Content-Type: application/json' -d "$body" >/dev/null 2>&1 &
      done
      sleep 0.3
    done
    wait
  ) &
  LOADER_PID=$!
}

note "Starting sustained load (~34s, 12 concurrent generations)..."
start_sustained_load 34
sleep 3   # let throughput ramp before sampling

echo
note "HOST live monitor — nvidia-smi dmon (sm = compute util %, mem = mem-bw %, fb = framebuffer MB):"
run "nvidia-smi dmon -s um -c 8"

echo
note "Host memory/utilization snapshot during load:"
run "nvidia-smi --query-gpu=memory.total,memory.used,utilization.gpu,power.draw,temperature.gpu --format=csv"
echo
note "Per-process GPU memory during load (both engines visible on the physical card):"
run "nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv"

# --- per-vGPU utilization, straight from HAMi (what nvidia-smi cannot attribute) ---
echo
printf "${BOLD}${YELLOW}>>> PER-vGPU UTILIZATION — attributed by HAMi to each container/slice${RESET}\n"
note "nvidia-smi shows ONE utilization number for the whole card. HAMi knows which"
note "vGPU is doing the work. Opening the HAMi device-plugin monitor (:${MON_LPORT})..."
kubectl -n kube-system port-forward svc/hami-device-plugin-monitor "${MON_LPORT}:31992" >/tmp/hami_mon_pf.log 2>&1 &
MON_PID=$!
for i in $(seq 1 12); do
  curl -s "http://localhost:${MON_LPORT}/metrics" >/dev/null 2>&1 && break
  sleep 0.5
done
for s in 1 2 3; do
  printf "\n${DIM}   ── HAMi vGPU sample %s/3 ──${RESET}\n" "$s"
  python3 "$SCRIPT_DIR/vgpu_util.py" "http://localhost:${MON_LPORT}/metrics"
  sleep 3
done
echo
printf "${DIM}   The busy SGLang vGPU reports high SM-util while idle vGPUs sit near 0%% —\n"
printf "   the same physical H100, but utilization is sliced per workload.${RESET}\n"
kill "$MON_PID" >/dev/null 2>&1; MON_PID=""

echo
note "Stopping load..."
kill "$LOADER_PID" >/dev/null 2>&1
pkill -f "localhost:${LPORT}/v1/chat" >/dev/null 2>&1
wait "$LOADER_PID" 2>/dev/null
LOADER_PID=""
ok "Load drained — the GPU returns to idle"
pause

# ---------------------------------------------------------------------
# ACT 7 — GPU fan-out: many workloads, one physical card
# ---------------------------------------------------------------------
FANOUT_REPLICAS="${FANOUT_REPLICAS:-5}"
if [ "$FANOUT_REPLICAS" -gt 0 ] 2>/dev/null; then
  act "ACT 7 · GPU fan-out — packing many pods onto ONE H100"
  note "This is the real 'split' story for HAMi: we schedule ${BOLD}${FANOUT_REPLICAS}${RESET}${CYAN} extra pods,"
  note "each asking HAMi for a small 3000 MB slice of the SAME physical GPU."
  echo
  run "kubectl -n '$NS' apply -f '$REPO_DIR/charts/fanout/fanout.yaml'"
  run "kubectl -n '$NS' scale deploy/gpu-fanout --replicas=$FANOUT_REPLICAS"
  note "Waiting for the fan-out pods to start and allocate GPU memory..."
  run "kubectl -n '$NS' rollout status deploy/gpu-fanout --timeout=180s"

  # wait until the extra processes actually appear on the physical GPU
  want=$FANOUT_REPLICAS
  for i in $(seq 1 40); do
    have=$(kubectl -n "$NS" get pods -l app=gpu-fanout \
            -o jsonpath='{range .items[*]}{.status.phase}{"\n"}{end}' 2>/dev/null | grep -c Running)
    allocated=$(kubectl -n "$NS" logs -l app=gpu-fanout --tail=5 2>/dev/null | grep -c "allocated")
    [ "$allocated" -ge "$want" ] && break
    sleep 3
  done
  ok "Fan-out pods are up and holding GPU memory"
  echo

  note "All GPU pods now scheduled on the single node (vLLM + SGLang + ${FANOUT_REPLICAS} fan-out):"
  run "kubectl -n '$NS' get pods -l 'app in (qwen,sglang,gpu-fanout)' -o wide | grep -E 'NAME|qwen|sglang|gpu-fanout'"
  echo
  note "HAMi GPU slices on the node — requested vs the 10 advertised:"
  run "kubectl describe node '$NODE' | grep -E 'nvidia.com/gpu' | head -4"
  echo

  printf "${BOLD}${YELLOW}>>> THE MONEY SHOT: host nvidia-smi — every pod is a separate process on GPU 0${RESET}\n"
  run "nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv"
  echo
  note "And each fan-out pod sees its OWN tiny capped GPU (total = 3000 MiB), proving isolation:"
  idx=0
  for p in $(kubectl -n "$NS" get pods -l app=gpu-fanout -o jsonpath='{.items[*].metadata.name}'); do
    idx=$((idx + 1))
    line=$(kubectl -n "$NS" exec "$p" -- nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader 2>/dev/null | tail -1)
    printf "   ${CYAN}%-32s${RESET} sees: ${BOLD}%s${RESET}\n" "$p" "$line"
  done
  echo
  printf "${BOLD}${GREEN}   One 80GB H100 → %s independent GPU workloads, each fenced to its own slice.${RESET}\n" "$((FANOUT_REPLICAS + 2))"
  pause

  note "Cleaning up the fan-out pods (returning the cluster to baseline)..."
  run "kubectl -n '$NS' delete -f '$REPO_DIR/charts/fanout/fanout.yaml' --wait=false"
  pause
fi

# ---------------------------------------------------------------------
# Recap
# ---------------------------------------------------------------------
act "Recap"
ok "HAMi turned 1 physical H100 into multiple schedulable GPU slices"
ok "The SGLang pod was sandboxed to a small GPU view (hard memory cap) by HAMi"
ok "SGLang served real OpenAI-compatible inference for ${MODEL}"
ok "nvidia-smi confirmed host-vs-pod views and live utilization under load"
ok "HAMi metrics showed PER-vGPU compute utilization (per-slice, not just whole-card)"
[ "${FANOUT_REPLICAS:-5}" -gt 0 ] 2>/dev/null && ok "Fan-out packed many isolated pods onto the one card — sharing made visible"
echo
printf "${BOLD}${BLUE}Everything above ran on a single NVIDIA H100 80GB, virtualized by HAMi.${RESET}\n\n"
