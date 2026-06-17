# kitunpacker -- KitOps init container.
# Pulls a ModelKit from an OCI registry (Jozu Hub by default) and unpacks the
# model weights into a shared volume for the inference engine to serve.
# Inspired by the Azure aks-byo-models-kaito lab, adapted for public Jozu Hub
# pulls (no login required) + automatic HF-layout flattening.
FROM alpine:3.20

RUN apk add --no-cache bash coreutils findutils ca-certificates curl tar \
    && curl -fsSL "https://github.com/jozu-ai/kitops/releases/latest/download/kitops-linux-x86_64.tar.gz" -o /tmp/kit.tgz \
    && tar -xzf /tmp/kit.tgz -C /usr/local/bin kit \
    && rm -f /tmp/kit.tgz \
    && kit version

# Defaults target the public Jozu Hub Qwen3-4B-Instruct ModelKit. Override any
# of these from the Pod spec to serve a different ModelKit / registry.
ENV MODELKIT_REF="jozu.ml/jonathangamer202002/qwen3-4b-instruct:latest" \
    UNPACK_PATH="/models" \
    MODEL_SUBDIR="qwen3"

COPY unpack.sh /usr/local/bin/unpack.sh
RUN chmod +x /usr/local/bin/unpack.sh

ENTRYPOINT ["/usr/local/bin/unpack.sh"]
