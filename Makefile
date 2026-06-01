# agentic-soccer — developer tasks
#
# Use `uv run --no-sync` everywhere: plain `uv run` re-syncs the venv and
# triggers a slow gfootball (C++) rebuild. gfootball is only available inside
# the Docker image, so host-side tests mock it.

.PHONY: test lint typecheck e2e-docker replay help

help:
	@echo "Targets:"
	@echo "  test         Run the test suite (host; gfootball mocked)"
	@echo "  lint         Run ruff over the repo"
	@echo "  typecheck    Run mypy over the repo"
	@echo "  e2e-docker   Full integration check in Docker (needs the built image)"
	@echo "  replay       Open the pygame replay viewer on match/replay.jsonl"

# Host unit + e2e tests. gfootball is mocked, so no Docker image required.
test:
	uv run --no-sync pytest -q

lint:
	uv run --no-sync ruff check .

typecheck:
	uv run --no-sync mypy .

# Full end-to-end integration check: runs the milestone harness inside the
# Docker image where gfootball is actually compiled and installed. This is the
# real integration gate — it requires the image to be built first
# (see Dockerfile / run_docker.sh / build_docker.txt).
e2e-docker:
	docker compose run --rm sim python run_milestones.py

# Replay a recorded match in the pygame viewer (needs a display).
replay:
	uv run --no-sync python -m replay.visualizer match/replay.jsonl
