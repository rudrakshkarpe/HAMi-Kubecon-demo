#!/usr/bin/env bash
# Build the KitOps/Jozu pipeline images and load them into the kind cluster.
#
#   ./build.sh                # build all three + kind load
#   ./build.sh kitunpacker    # build just the init image + load
#
# Images produced (local-only, consumed via imagePullPolicy: IfNotPresent):
#   hami-kitunpacker:latest   alpine + kit CLI + unpack/flatten script
#   hami-vllm-jozu:latest     vllm/vllm-openai:latest + custom local-serve cmd
#   hami-sglang-jozu:latest   lmsysorg/sglang:latest + custom local-serve cmd
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KIND_CLUSTER="${KIND_CLUSTER:-hami-demo}"
# docker may require group context on this host: `sg docker -c '...'`
DOCKER="${DOCKER:-docker}"
KIND="${KIND:-kind}"

build_and_load() {
  local name="$1" ctx="$2"
  echo "==> building ${name}:latest"
  ${DOCKER} build -t "${name}:latest" "${ctx}"
  echo "==> loading ${name}:latest into kind/${KIND_CLUSTER}"
  ${KIND} load docker-image "${name}:latest" --name "${KIND_CLUSTER}"
}

target="${1:-all}"
case "${target}" in
  kitunpacker|all) build_and_load hami-kitunpacker "${HERE}/docker/kitunpacker" ;;
esac
case "${target}" in
  vllm|all)        build_and_load hami-vllm-jozu   "${HERE}/docker/vllm" ;;
esac
case "${target}" in
  sglang|all)      build_and_load hami-sglang-jozu "${HERE}/docker/sglang" ;;
esac

echo "==> done. images in kind node:"
${DOCKER} exec "${KIND_CLUSTER}-control-plane" crictl images 2>/dev/null | grep -E 'hami-(kitunpacker|vllm-jozu|sglang-jozu)' || true
