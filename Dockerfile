FROM python:3.14-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /opt/app-root/src

COPY pyproject.toml .
COPY src ./src
COPY scripts ./scripts
COPY ogx-config.yaml .
COPY entrypoint.sh .

ENV VIRTUAL_ENV=/opt/app-root/.venv
ENV PATH="/opt/app-root/.venv/bin:$PATH"
RUN uv venv /opt/app-root/.venv \
    && uv pip install --no-cache -e . \
        "ogx[starter]" openai botocore chardet sqlite-vec pypdf markitdown \
    && python scripts/patch_ogx_streaming.py \
    && chmod +x /opt/app-root/src/entrypoint.sh

EXPOSE 8321 8888 8889 8890 8891 8892 8893

ENTRYPOINT ["/opt/app-root/src/entrypoint.sh"]
