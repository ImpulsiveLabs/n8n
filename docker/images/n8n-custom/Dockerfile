# Stage 1: Python Base with Dependencies
FROM --platform=linux/amd64 python:3.13-slim AS python_base

# Install build dependencies to fix hdbscan build issue
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libc-dev libffi-dev liblapack-dev libblas-dev gfortran \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install required Python packages
RUN pip install --no-cache-dir --upgrade pip setuptools wheel tqdm pyalex sentence-transformers numpy scikit-learn hdbscan

# Stage 2: Node.js and N8N Build Stage
FROM --platform=linux/amd64 n8nio/base:20 AS builder

# Install Python and tools for swap
RUN apk add --no-cache python3 py3-pip py3-virtualenv dpkg && \
    [ ! -e /usr/bin/python ] || rm /usr/bin/python && \
    ln -s /usr/bin/python3 /usr/bin/python

WORKDIR /src
COPY . /src

# Install dependencies with pnpm (caching included)
RUN --mount=type=cache,id=pnpm-store,target=/root/.local/share/pnpm/store \
    --mount=type=cache,id=pnpm-metadata,target=/root/.cache/pnpm/metadata \
    DOCKER_BUILD=true pnpm install --frozen-lockfile

# Build and postbuild
RUN pnpm build
RUN pnpm postbuild

# Clean up package.json and remove dev dependencies
RUN jq 'del(.pnpm.patchedDependencies)' package.json > package.json.tmp && mv package.json.tmp package.json
RUN node .github/scripts/trim-fe-packageJson.js

# Remove source files and typings
RUN find . -type f -name "*.ts" -o -name "*.js.map" -o -name "*.vue" -o -name "tsconfig.json" -o -name "*.tsbuildinfo" | xargs rm -rf

# Deploy production n8n package
RUN mkdir /compiled
RUN NODE_ENV=production DOCKER_BUILD=true pnpm --filter=n8n --prod --no-optional --legacy deploy /compiled

# Stage 3: Final Image for Production
FROM python:3.13-slim AS final_image

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl gnupg lsb-release tini wget \
    gcc g++ libc-dev libffi-dev liblapack-dev libblas-dev gfortran \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Node.js 20.x
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs && apt-get clean

# Install required Python packages
RUN pip install --no-cache-dir tqdm pyalex sentence-transformers numpy scikit-learn hdbscan

# Set up working directory and copy compiled n8n
WORKDIR /home/node
COPY --from=builder /compiled /usr/local/lib/node_modules/n8n
COPY docker/images/n8n/docker-entrypoint.sh /

# Install Task Runner Launcher
ARG TARGETPLATFORM
ARG LAUNCHER_VERSION=1.1.0
COPY docker/images/n8n/n8n-task-runners.json /etc/n8n-task-runners.json
RUN if [ "$TARGETPLATFORM" = "linux/amd64" ]; then export ARCH_NAME="amd64"; \
    elif [ "$TARGETPLATFORM" = "linux/arm64" ]; then export ARCH_NAME="arm64"; fi; \
    mkdir /launcher-temp && cd /launcher-temp && \
    wget https://github.com/n8n-io/task-runner-launcher/releases/download/${LAUNCHER_VERSION}/task-runner-launcher-${LAUNCHER_VERSION}-linux-${ARCH_NAME}.tar.gz && \
    wget https://github.com/n8n-io/task-runner-launcher/releases/download/${LAUNCHER_VERSION}/task-runner-launcher-${LAUNCHER_VERSION}-linux-${ARCH_NAME}.tar.gz.sha256 && \
    echo "$(cat task-runner-launcher-${LAUNCHER_VERSION}-linux-${ARCH_NAME}.tar.gz.sha256) task-runner-launcher-${LAUNCHER_VERSION}-linux-${ARCH_NAME}.tar.gz" > checksum.sha256 && \
    sha256sum -c checksum.sha256 && \
    tar xvf task-runner-launcher-${LAUNCHER_VERSION}-linux-${ARCH_NAME}.tar.gz --directory=/usr/local/bin && \
    cd - && rm -r /launcher-temp

# Rebuild sqlite3 and set up paths
RUN cd /usr/local/lib/node_modules/n8n && \
    npm rebuild sqlite3 && \
    ln -s /usr/local/lib/node_modules/n8n/bin/n8n /usr/local/bin/n8n && \
    mkdir -p /home/node/.n8n && \
    chown -R root:root /home/node/.n8n /usr/local/lib/node_modules/n8n && \
    chmod -R 755 /home/node/.n8n /usr/local/lib/node_modules/n8n

# Final environment setup
ENV SHELL /bin/sh
ENTRYPOINT ["tini", "--", "/docker-entrypoint.sh"]
