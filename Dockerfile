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
# VERIFIED: this Dockerfile built agentic-soccer:latest (2.74GB) with
# gfootball-2.10.2 compiled and installed, DOCKER_BUILD2_EXIT=0.
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
# gfootball — the slow C++ engine compile (verified to succeed at the default
# parallelism on an 8-CPU / 7.75 GiB Docker VM).
#
# --no-build-isolation is REQUIRED: gfootball's setup.py imports psutil/numpy
# during the build; pip's default isolated build env lacks them ->
# "ModuleNotFoundError: No module named 'psutil'". We pre-install them and
# disable isolation so the build sees them. Do NOT pip-install the `cmake`
# wheel: it is 4.x and would shadow apt cmake 3.28, reviving the CMake-4
# "cmake_minimum_required < 3.5 removed" policy error.
# ---------------------------------------------------------------------------
RUN pip install --upgrade pip setuptools wheel \
    && pip install psutil "numpy>=1.26" \
    && pip install --no-build-isolation gfootball==2.10.2

# ---------------------------------------------------------------------------
# Two runtime deps gfootball 2.10.2 needs but does NOT declare correctly:
#
#  * six — gfootball imports `from six.moves import range/zip/cPickle` in ~9
#    modules but omits six from install_requires -> ModuleNotFoundError.
#
#  * gym==0.22.0 — gfootball declares `gym>=0.11.0` (unpinned) so pip pulls
#    gym 0.26.2, whose Env wrapper does `obs, info = env.reset()`. gfootball
#    2.10.2 returns a SINGLE-value reset(), so the real engine's
#    `obs_list = env.reset()` raises "too many values to unpack (expected 2)".
#    gym must be downgraded to a single-return-reset version. 0.21.0 is the
#    classic target but its ancient sdist fails to build on this image's
#    Python 3.12 / modern setuptools. 0.22.0 is the OLDEST gym that BOTH
#    installs cleanly here (no setuptools<66 dance) AND keeps the old
#    single-value reset() gfootball needs. VERIFIED end-to-end in-container
#    (simulator-agent): reset (72,96,4) / GFOOTBALL_WORKS / full 500-tick
#    11v11 match. Runs against the image's numpy 2.4.6 — do NOT pin numpy<2
#    (would risk the gfootball _gameplayfootball.so ABI built vs 2.4.6).
# ---------------------------------------------------------------------------
RUN pip install "gym==0.22.0" six \
        "fastmcp>=2.0" "anthropic>=0.40" "pygame>=2.6" requests

# Project code (see .dockerignore for exclusions). match/ is a mounted volume.
COPY simulator/ ./simulator/
COPY mcp_server/ ./mcp_server/
COPY agents/ ./agents/
COPY replay/ ./replay/
COPY tests/ ./tests/
COPY check_milestones.py main.py docker_entry.py run_milestones.py ./
RUN mkdir -p /app/match

EXPOSE 8765

# Default: prove gfootball imports + steps headless inside the container.
# Override with `docker run ... python docker_entry.py` to run the full
# match + MCP server (see docker_entry.py / README).
CMD ["python", "tests/verify_gfootball.py"]
