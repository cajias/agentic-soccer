# syntax=docker/dockerfile:1
#
# gfootball (Google Research Football) runs cleanly in Linux — this image is
# how we run the REAL engine after the native macOS arm64 build proved
# intractable (Boost.Python is ABI-locked to the build interpreter and Homebrew
# ships no libboost_python312; CMake 4.x rejects gfootball's old policy).
#
# ubuntu:24.04 is chosen deliberately — it makes all three macOS walls vanish:
#   * native python3 IS 3.12  -> no deadsnakes, interpreter matches the project
#   * apt libboost-all-dev (1.83) is built against that same python3.12
#     -> Boost.Python component `python312` resolves (the macOS blocker)
#   * apt cmake (3.28) still ships FindBoost and predates the CMake-4 policy
#     removal -> no boost_system / cmake_minimum_required errors
#
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    SDL_VIDEODRIVER=dummy \
    SDL_AUDIODRIVER=dummy \
    MPLBACKEND=Agg

# ---------------------------------------------------------------------------
# System deps for gfootball's C++ engine (SDL2 stack + Boost + OpenGL), per
# the gfootball README. cmake/build-essential drive the in-tree CMake build.
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        cmake \
        build-essential \
        python3 \
        python3-dev \
        python3-venv \
        python3-pip \
        libgl1-mesa-dev \
        libsdl2-dev \
        libsdl2-image-dev \
        libsdl2-ttf-dev \
        libsdl2-gfx-dev \
        libboost-all-dev \
        libdirectfb-dev \
        libst-dev \
        mesa-utils \
    && rm -rf /var/lib/apt/lists/*

# Isolated venv (avoids PEP 668 "externally managed" and keeps the same
# python3.12 ABI the apt Boost.Python was built against).
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# ---------------------------------------------------------------------------
# gfootball as its own cached layer (the slow ~several-minute C++ compile).
# psutil + numpy are imported by gfootball's setup.py / build_game_engine.sh
# during the build. They are installed FIRST, then gfootball is built with
# --no-build-isolation so its setup.py can see them: pip's default build
# isolation runs the build in a fresh env WITHOUT psutil/numpy, which fails with
# "ModuleNotFoundError: No module named 'psutil'". Do NOT pip-install the
# `cmake` wheel here: it would be CMake 4.x and shadow the apt cmake 3.28.
# ---------------------------------------------------------------------------
RUN pip install --upgrade pip setuptools wheel \
    && pip install psutil "numpy>=1.26" \
    && pip install --no-build-isolation gfootball==2.10.2

# Project runtime deps.
RUN pip install "fastmcp>=2.0" "anthropic>=0.40" "pygame>=2.6" requests

# Project code (see .dockerignore for exclusions). match/ is a mounted volume.
COPY simulator/ ./simulator/
COPY mcp_server/ ./mcp_server/
COPY agents/ ./agents/
COPY replay/ ./replay/
COPY tests/ ./tests/
COPY check_milestones.py main.py docker_entry.py ./
RUN mkdir -p /app/match

EXPOSE 8765

# Default: prove gfootball imports + steps headless inside the container.
# Override with `docker run ... python docker_entry.py` to run the full
# match + MCP server (see docker_entry.py / README).
CMD ["python", "tests/verify_gfootball.py"]
