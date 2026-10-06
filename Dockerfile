# vLLM 0.28 + the vendored clef-vllm plugin. The build context is ./vllm_plugin only:
#   docker build -t clef-vllm:0.28.0 -f Dockerfile vllm_plugin
FROM vllm/vllm-openai:v0.28.0
COPY . /tmp/clef-vllm-plugin
RUN pip install --no-deps --no-cache-dir --root-user-action=ignore /tmp/clef-vllm-plugin && rm -rf /tmp/clef-vllm-plugin
ENTRYPOINT ["clef-systemone"]
