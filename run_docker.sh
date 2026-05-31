#!/usr/bin/env bash
# Build and run the agentic-soccer container (gfootball engine + MCP server).
#
# Usage:
#   ./run_docker.sh build     # build the image
#   ./run_docker.sh verify    # build (if needed) + prove gfootball runs in-container
#   ./run_docker.sh up        # build + run match + MCP server (foreground)
#   ./run_docker.sh           # same as: build then verify
set -euo pipefail
cd "$(dirname "$0")"

IMAGE="agentic-soccer:latest"

build() {
  echo ">> Building $IMAGE (first build compiles gfootball's C++ engine; ~several min)"
  docker build -t "$IMAGE" .
}

verify() {
  echo ">> Verifying gfootball imports + steps INSIDE the container"
  docker run --rm -e SDL_VIDEODRIVER=dummy "$IMAGE" python tests/verify_gfootball.py
}

up() {
  echo ">> Starting match + MCP server (port 8765); replay.jsonl -> ./match/"
  mkdir -p match
  docker run --rm -p 8765:8765 -v "$(pwd)/match:/app/match" \
    -e SDL_VIDEODRIVER=dummy -e MCP_HOST=0.0.0.0 \
    -e ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-}" \
    "$IMAGE" python docker_entry.py
}

cmd="${1:-default}"
case "$cmd" in
  build)  build ;;
  verify) build && verify ;;
  up)     build && up ;;
  default) build && verify ;;
  *) echo "usage: $0 {build|verify|up}" >&2; exit 2 ;;
esac
