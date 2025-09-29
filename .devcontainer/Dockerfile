FROM harbor.apmic.ai/library/python311

# Install uv for faster dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Caddy
COPY .devcontainer/Caddyfile /etc/caddy/Caddyfile

# Hasura
RUN curl -sL https://github.com/hasura/graphql-engine/raw/stable/cli/get.sh | VERSION=v2.33.0 bash

# Set up workspace permissions
RUN PROJECT_PATH=$(find /workspaces -maxdepth 1 -type d | grep -v "^/workspaces$" | grep -v "coder" | head -n 1) && \
    cd "${PROJECT_PATH}" && \
    uv sync

ENV PATH="${PROJECT_PATH}/.venv/bin:$PATH"

# Install VS Code extensions
COPY .vscode/extensions.json extensions.json
RUN awk '/"recommendations"/,/\]/{print}' extensions.json \
    | grep -o '".*\..*"' \
    | tr -d '"' \
    | xargs -I {} code --extensions-dir "${HOME}/.vscode-server/extensions" --install-extension {} \
    ; exit 0

COPY requirements.txt requirements.txt
RUN pip3 install --upgrade pip setuptools \
    && pip3 install -r requirements.txt

