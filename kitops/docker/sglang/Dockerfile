# Custom SGLang image for the KitOps / Jozu Hub pipeline.
# The model is delivered at runtime by the kitunpacker initContainer; this image
# only bakes the custom serve command that loads the model from a local dir.
FROM lmsysorg/sglang:latest

ENV MODEL_DIR="/models/qwen3" \
    SERVED_NAME="qwen3-4b-instruct" \
    CONTEXT_LEN="8192" \
    MEM_FRACTION="0.8" \
    PORT="30000" \
    ATTENTION_BACKEND="triton"

COPY serve.sh /usr/local/bin/serve.sh
RUN chmod +x /usr/local/bin/serve.sh

# Override the stock launch_server entrypoint with our local-model command.
ENTRYPOINT ["/usr/local/bin/serve.sh"]
