# syntax=docker/dockerfile:1.4
ARG NODE_VERSION=22.21.1

# ============================================================
# Stage 1: n8n build stage (Debian)
# ============================================================
FROM --platform=linux/amd64 node:22 AS builder

WORKDIR /src
COPY . /src

# jq needed for package.json transform
RUN apt-get update && apt-get install -y --no-install-recommends jq \
  && rm -rf /var/lib/apt/lists/*

RUN corepack enable && corepack prepare pnpm@9 --activate

RUN --mount=type=cache,id=pnpm-store,target=/root/.local/share/pnpm/store \
    --mount=type=cache,id=pnpm-metadata,target=/root/.cache/pnpm/metadata \
    DOCKER_BUILD=true pnpm install --frozen-lockfile

RUN pnpm build

RUN jq 'del(.pnpm.patchedDependencies)' package.json > package.json.tmp && mv package.json.tmp package.json
RUN node .github/scripts/trim-fe-packageJson.js

RUN find . \( -name "*.ts" -o -name "*.js.map" -o -name "*.vue" -o -name "tsconfig.json" -o -name "*.tsbuildinfo" \) \
    -type f -print0 | xargs -0 rm -f

RUN mkdir /compiled && \
    NODE_ENV=production DOCKER_BUILD=true pnpm --filter=n8n --prod --no-optional --legacy deploy /compiled


# ============================================================
# Stage 2: final runtime image (Alpine)
# ============================================================
FROM node:${NODE_VERSION}-alpine AS final_image

# Install runtime deps + tools needed for launcher download + checksum verify
# coreutils => sha256sum, wget => download
RUN apk add --no-cache \
    bash ca-certificates curl tini tzdata \
    git openssh jq graphicsmagick \
    wget coreutils

# If you want full ICU like your previous base
RUN npm i -g full-icu@1.5.0
ENV NODE_ICU_DATA=/usr/local/lib/node_modules/full-icu

WORKDIR /home/node

# n8n runtime
COPY --from=builder /compiled /usr/local/lib/node_modules/n8n

# scripts + extras
COPY docker/images/n8n/docker-entrypoint.sh /
COPY docker/images/runners/n8n-task-runners.json /etc/n8n-task-runners.json
COPY extras/ /home/node/extras

# task-runner launcher (wget + sha256sum now exist)
ARG TARGETPLATFORM
ARG LAUNCHER_VERSION=1.1.0
RUN set -eux; \
    if [ "$TARGETPLATFORM" = "linux/amd64" ]; then ARCH_NAME="amd64"; \
    elif [ "$TARGETPLATFORM" = "linux/arm64" ]; then ARCH_NAME="arm64"; \
    else echo "Unsupported TARGETPLATFORM=$TARGETPLATFORM" && exit 1; fi; \
    mkdir /launcher-temp && cd /launcher-temp; \
    wget -q https://github.com/n8n-io/task-runner-launcher/releases/download/${LAUNCHER_VERSION}/task-runner-launcher-${LAUNCHER_VERSION}-linux-${ARCH_NAME}.tar.gz; \
    wget -q https://github.com/n8n-io/task-runner-launcher/releases/download/${LAUNCHER_VERSION}/task-runner-launcher-${LAUNCHER_VERSION}-linux-${ARCH_NAME}.tar.gz.sha256; \
    echo "$(cat task-runner-launcher-${LAUNCHER_VERSION}-linux-${ARCH_NAME}.tar.gz.sha256) task-runner-launcher-${LAUNCHER_VERSION}-linux-${ARCH_NAME}.tar.gz" > checksum.sha256; \
    sha256sum -c checksum.sha256; \
    tar xvf task-runner-launcher-${LAUNCHER_VERSION}-linux-${ARCH_NAME}.tar.gz --directory=/usr/local/bin; \
    cd /; rm -rf /launcher-temp

# final setup
RUN cd /usr/local/lib/node_modules/n8n && \
    npm rebuild sqlite3 && \
    ln -s /usr/local/lib/node_modules/n8n/bin/n8n /usr/local/bin/n8n && \
    mkdir -p /home/node/.n8n && \
    chown -R root:root /home/node/.n8n /usr/local/lib/node_modules/n8n && \
    chmod -R 755 /home/node/.n8n /usr/local/lib/node_modules/n8n

EXPOSE 5678
ENTRYPOINT ["tini", "--", "/docker-entrypoint.sh"]
