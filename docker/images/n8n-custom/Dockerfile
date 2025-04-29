# syntax=docker/dockerfile:1.4

# Stage 1: Node.js and N8N Build Stage
FROM --platform=linux/amd64 n8nio/base:20 AS builder

WORKDIR /src
COPY . /src

# Enable Corepack and activate a pnpm version
RUN corepack enable && corepack prepare pnpm@9 --activate

# Install dependencies with pnpm (caching included)
RUN --mount=type=cache,id=pnpm-store,target=/root/.local/share/pnpm/store \
    --mount=type=cache,id=pnpm-metadata,target=/root/.cache/pnpm/metadata \
    DOCKER_BUILD=true pnpm install --frozen-lockfile

RUN pnpm build

# (optional) jq is not guaranteed to be present in this base image; install if needed
# RUN apt-get update && apt-get install -y --no-install-recommends jq && rm -rf /var/lib/apt/lists/*

RUN jq 'del(.pnpm.patchedDependencies)' package.json > package.json.tmp && mv package.json.tmp package.json
RUN node .github/scripts/trim-fe-packageJson.js

RUN find . \( -name "*.ts" -o -name "*.js.map" -o -name "*.vue" -o -name "tsconfig.json" -o -name "*.tsbuildinfo" \) -type f -print0 | xargs -0 rm -f

RUN mkdir /compiled && \
    NODE_ENV=production DOCKER_BUILD=true pnpm --filter=n8n --prod --no-optional --legacy deploy /compiled

# Stage 2: Final Image for Production
FROM --platform=linux/amd64 node:20-slim AS final_image
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl gnupg lsb-release ca-certificates tini wget \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /home/node
COPY --from=builder /compiled /usr/local/lib/node_modules/n8n
COPY docker/images/n8n/docker-entrypoint.sh /

ARG TARGETPLATFORM
ARG LAUNCHER_VERSION=1.1.2
COPY docker/images/n8n/n8n-task-runners.json /etc/n8n-task-runners.json
COPY extras/ /home/node/extras
RUN if [ "$TARGETPLATFORM" = "linux/amd64" ]; then ARCH_NAME="amd64"; \
    elif [ "$TARGETPLATFORM" = "linux/arm64" ]; then ARCH_NAME="arm64"; fi; \
    mkdir /launcher-temp && cd /launcher-temp && \
    wget https://github.com/n8n-io/task-runner-launcher/releases/download/${LAUNCHER_VERSION}/task-runner-launcher-${LAUNCHER_VERSION}-linux-${ARCH_NAME}.tar.gz && \
    wget https://github.com/n8n-io/task-runner-launcher/releases/download/${LAUNCHER_VERSION}/task-runner-launcher-${LAUNCHER_VERSION}-linux-${ARCH_NAME}.tar.gz.sha256 && \
    echo "$(cat task-runner-launcher-${LAUNCHER_VERSION}-linux-${ARCH_NAME}.tar.gz.sha256) task-runner-launcher-${LAUNCHER_VERSION}-linux-${ARCH_NAME}.tar.gz" > checksum.sha256 && \
    sha256sum -c checksum.sha256 && \
    tar xvf task-runner-launcher-${LAUNCHER_VERSION}-linux-${ARCH_NAME}.tar.gz --directory=/usr/local/bin && \
    cd - && rm -r /launcher-temp

RUN cd /usr/local/lib/node_modules/n8n && \
    npm rebuild sqlite3 && \
    ln -s /usr/local/lib/node_modules/n8n/bin/n8n /usr/local/bin/n8n && \
    mkdir -p /home/node/.n8n && \
    chown -R root:root /home/node/.n8n /usr/local/lib/node_modules/n8n && \
    chmod -R 755 /home/node/.n8n /usr/local/lib/node_modules/n8n

ENV SHELL=/bin/sh
ENTRYPOINT ["tini", "--", "/docker-entrypoint.sh"]
