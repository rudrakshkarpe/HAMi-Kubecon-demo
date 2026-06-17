# Custom vLLM image for the KitOps / Jozu Hub pipeline.
# The model is NOT baked in -- it is delivered at runtime by the kitunpacker
# initContainer. This image only bakes the custom serve command that loads the
# model from a local directory instead of pulling from Hugging Face.
FROM vllm/vllm-openai:latest

ENV MODEL_DIR="/models/qwen3" \
    SERVED_NAME="qwen3-4b-instruct" \
    MAX_MODEL_LEN="8192" \
    GPU_MEM_UTIL="0.85" \
    PORT="8000"

COPY serve.sh /usr/local/bin/serve.sh
RUN chmod +x /usr/local/bin/serve.sh

# Override the stock `vllm serve <hf-repo>` entrypoint with our local-model command.
ENTRYPOINT ["/usr/local/bin/serve.sh"]
