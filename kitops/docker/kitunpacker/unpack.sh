#!/usr/bin/env sh
# kitunpacker: pull a ModelKit from an OCI registry (Jozu Hub by default) and
# unpack the model into a Hugging Face-style directory that vLLM / SGLang can
# load directly -- no Hugging Face download at runtime.
#
# Env (all overridable from the Pod spec):
#   MODELKIT_REF   full ModelKit reference, e.g. jozu.ml/<org>/<repo>:<tag>
#   UNPACK_PATH    root volume to unpack into                 (default /models)
#   MODEL_SUBDIR   final model dir under UNPACK_PATH          (default qwen3)
#   REGISTRY_URL/USERNAME/PASSWORD  optional creds for PRIVATE registries
set -eu

MODELKIT_REF="${MODELKIT_REF:?MODELKIT_REF is required}"
UNPACK_PATH="${UNPACK_PATH:-/models}"
MODEL_SUBDIR="${MODEL_SUBDIR:-qwen3}"
DEST="${UNPACK_PATH}/${MODEL_SUBDIR}"
RAW="${UNPACK_PATH}/.raw-${MODEL_SUBDIR}"
LOCK="${UNPACK_PATH}/.lock-${MODEL_SUBDIR}"

# keep the kit pull cache on the (large) mounted volume, not the tiny rootfs
export KITOPS_HOME="${UNPACK_PATH}/.kitcache"

ready() { [ -f "${DEST}/config.json" ] && ls "${DEST}"/*.safetensors >/dev/null 2>&1; }

echo "[kitunpacker] ref=${MODELKIT_REF} -> ${DEST}"

if ready; then
  echo "[kitunpacker] model already present, skipping unpack"
  exit 0
fi

# simple cross-pod lock (mkdir is atomic) so two engines sharing one cache
# don't race to write the same files.
if ! mkdir "${LOCK}" 2>/dev/null; then
  echo "[kitunpacker] another unpack in progress, waiting for it to finish..."
  i=0
  while [ "${i}" -lt 360 ]; do
    ready && { echo "[kitunpacker] model became ready"; exit 0; }
    i=$((i + 1)); sleep 5
  done
  echo "[kitunpacker] timed out waiting for peer unpack" >&2
  exit 1
fi
# shellcheck disable=SC2064
trap "rmdir '${LOCK}' 2>/dev/null || true" EXIT INT TERM

# optional login for private registries (public Jozu Hub needs none)
if [ -n "${REGISTRY_URL:-}" ] && [ -n "${USERNAME:-}" ] && [ -n "${PASSWORD:-}" ]; then
  echo "[kitunpacker] logging in to ${REGISTRY_URL} as ${USERNAME}"
  echo "${PASSWORD}" | kit login "${REGISTRY_URL}" -u "${USERNAME}" --password-stdin
fi

rm -rf "${RAW}"; mkdir -p "${RAW}"
echo "[kitunpacker] pulling + unpacking model layers from registry..."
kit unpack --filter model "${MODELKIT_REF}" -d "${RAW}"

# Flatten: ModelKits may store the .safetensors shards in a model/ subdir while
# config.json / *.index.json / tokenizer sit one level up. vLLM/transformers
# need them all in one directory, so collect everything into DEST.
SRC_CFG="$(find "${RAW}" -name config.json | head -1)"
[ -n "${SRC_CFG}" ] || { echo "[kitunpacker] config.json not found after unpack" >&2; exit 1; }
SRC="$(dirname "${SRC_CFG}")"

mkdir -p "${DEST}"
# all weight shards, wherever they live under the unpacked tree
find "${SRC}" -name '*.safetensors' -exec mv -f {} "${DEST}/" \;
# all top-level metadata files (config, index, tokenizer, vocab, generation cfg)
find "${SRC}" -maxdepth 1 -type f -exec mv -f {} "${DEST}/" \;

rm -rf "${RAW}" "${KITOPS_HOME}"

echo "[kitunpacker] final model directory:"
ls -la "${DEST}"
ready || { echo "[kitunpacker] validation failed: missing config or shards" >&2; exit 1; }
echo "[kitunpacker] done."
