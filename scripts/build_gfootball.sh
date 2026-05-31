#!/bin/bash
# Atomic gfootball build + headless verify for macOS arm64 / Python 3.12.
# Installs the required Homebrew C++ deps, then runs the REAL reproducible
# install path (`uv sync`) teammates will use, then verifies headless import
# in a single command so a concurrent venv rebuild can't split build from test.
#
# Root causes handled (discovered 2026-05-31):
#   1. uv build isolation hides psutil/numpy/setuptools that gfootball's
#      setup.py imports          -> [tool.uv.extra-build-dependencies] in pyproject
#   2. Homebrew SDL2/boost/glew headers+libs
#                                -> [tool.uv.extra-build-variables] CMAKE_PREFIX_PATH etc.
#   3. CMake 4.x rejects engine's cmake_minimum_required(VERSION 3.4)
#                                -> CMAKE_POLICY_VERSION_MINIMUM=3.5 (in extra-build-variables)
#   4. Engine CMakeLists REQUIRES SDL2 satellite libs not in the original setup
#                                -> brew install sdl2_image sdl2_ttf sdl2_gfx (below)
set -uo pipefail
cd "$(dirname "$0")/.."

export SDL_VIDEODRIVER=dummy
export SDL_AUDIODRIVER=dummy
export MPLBACKEND=Agg

echo "### STEP 0: ensure required Homebrew C++ deps are installed"
brew install cmake sdl2 sdl2_image sdl2_ttf sdl2_gfx boost glew 2>&1 | tail -8
echo "--- installed versions ---"
brew list --versions sdl2 sdl2_image sdl2_ttf sdl2_gfx boost glew cmake 2>&1

echo "### STEP 1: clean reproducible install via uv sync (the real path)"
uv sync --extra dev --preview-features extra-build-dependencies,extra-build-variables 2>&1 | tail -30
echo "SYNC_EXIT=${PIPESTATUS[0]}"

echo "### STEP 2: headless import + reset + step"
uv run python tests/verify_gfootball.py 2>&1 | tail -15
echo "VERIFY_EXIT=${PIPESTATUS[0]}"
echo "### DONE"
