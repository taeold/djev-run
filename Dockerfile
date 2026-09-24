FROM docker.io/vllm/vllm-openai:nightly

COPY server.py snake.html dino.html tetris.html /opt/dgemma/
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENV LD_LIBRARY_PATH="/usr/local/cuda/compat"
EXPOSE 8080
ENTRYPOINT ["/entrypoint.sh"]
