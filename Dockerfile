ARG BASE=vllm/vllm-openai:nightly-dee37d89115db4c94a820a79a78a7828e141c910

FROM alpine/git:latest AS fork
ARG VLLM_FORK=https://github.com/mmastrac/vllm.git
ARG VLLM_UPSTREAM=https://github.com/vllm-project/vllm.git
ARG VLLM_REF=6591b093b29536dd070c6af3628b734025c53e23
ARG VLLM_BASE=dee37d89115db4c94a820a79a78a7828e141c910
ARG STRUCT_REPO=https://github.com/razorback16/vllm.git
ARG STRUCT_BRANCH=structured-reads-54309
RUN git clone --filter=blob:none --quiet "${VLLM_FORK}" /fork \
    && cd /fork \
    && git checkout --quiet "${VLLM_REF}" \
    && git fetch --filter=blob:none --quiet "${STRUCT_REPO}" "${STRUCT_BRANCH}" \
    && git -c user.name=build -c user.email=build@local cherry-pick 5737ace46042ea3ad52505d3c6daf32b87854692 \
    && git fetch --filter=blob:none --quiet "${VLLM_UPSTREAM}" "${VLLM_BASE}" \
    && mb=$(git merge-base "${VLLM_BASE}" HEAD) \
    && git diff --name-only "$mb" HEAD -- vllm > /fork/changed.txt \
    && git diff --name-only --diff-filter=A "$mb" HEAD -- vllm > /fork/added.txt \
    && cat /fork/changed.txt

FROM ${BASE}
ARG VLLM_BASE=dee37d89115db4c94a820a79a78a7828e141c910

COPY patches/link_cuda_headers.sh /tmp/link_cuda_headers.sh
RUN chmod +x /tmp/link_cuda_headers.sh && /tmp/link_cuda_headers.sh && rm /tmp/link_cuda_headers.sh

COPY --from=fork /fork /tmp/fork
COPY patches/overlay_vllm.py patches/raise_recompile_limit.py /tmp/
RUN python3 /tmp/overlay_vllm.py /tmp/fork "${VLLM_BASE}" && \
    python3 /tmp/raise_recompile_limit.py && \
    rm -rf /tmp/fork /tmp/overlay_vllm.py /tmp/raise_recompile_limit.py

# Stage tokenizer files into /models/dgemma for build-time unit test verification
RUN mkdir -p /models/dgemma && \
    for f in chat_template.jinja config.json generation_config.json processor_config.json tokenizer.json tokenizer_config.json; do \
      curl -sfL "https://huggingface.co/nvidia/diffusiongemma-26B-A4B-it-NVFP4/resolve/main/${f}" -o "/models/dgemma/${f}"; \
    done

COPY server/structured_server.py server/playground.html server/test_structured_server.py /opt/dgemma/
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh && \
    python3 -m py_compile /opt/dgemma/structured_server.py && \
    SERVER_DIR=/opt/dgemma TOKENIZER=/models/dgemma CHAT_TEMPLATE=/models/dgemma/chat_template.jinja python3 /opt/dgemma/test_structured_server.py

ENV PORT=8080 \
    MODEL=/mnt/gcs/dgemma \
    CANVAS=128 \
    MAX_SEQS=32 \
    MAX_MODEL_LEN=4096 \
    GPU_UTIL=0.40 \
    KV_CACHE_GB=2 \
    ATTN=TRITON_ATTN \
    COPY_TO_SHM=1 \
    TEST_PAGE=1 \
    VLLM_UF_EAGER_ALL=1 \
    VLLM_FLASHINFER_MOE_BACKEND=masked_gemm \
    VLLM_WORKER_MULTIPROC_METHOD=spawn \
    CUDA_MODULE_LOADING=LAZY

EXPOSE 8080
ENTRYPOINT ["/entrypoint.sh"]
