# gfootball on macOS arm64 + CPython 3.12 — escape-hatch research

Date: 2026-05-31. Research only; nothing installed.

NOTE on method: every load-bearing fact was confirmed via authoritative
single-purpose GitHub/PyPI/Homebrew endpoints (per-issue `/issues/{n}`,
`/repos/{owner}/{repo}`, `/git/refs`, formulae.brew.sh). One self-inflicted
error to be transparent about: I first queried the fork as
`troyedwardsjr/football` (that's the PR *author's username*) and got 404, then
briefly concluded the fork was fabricated. That was MY mistake — the fork's
actual owner is `trygentic`. `trygentic/football` returns 200 and is a real fork
of google-research/football. Corrected below.

## 1. Prebuilt wheel / conda package — DOES NOT EXIST for arm64+py3.12
- PyPI `gfootball` ships ONLY source dists: filenames are all `*.tar.gz`
  (1.0 … 2.10.2) plus one `2.10.2.zip`. There is ZERO `.whl` of any kind,
  hence no macOS arm64 wheel. (https://pypi.org/pypi/gfootball/json)
- anaconda.org search `?name=gfootball` returns `[]` (empty).
  (https://api.anaconda.org/search?name=gfootball)
- conda-forge has no `gfootball` package:
  https://api.anaconda.org/package/conda-forge/gfootball -> HTTP 404.
- Conclusion: `conda install -c conda-forge gfootball` and any
  `pip install gfootball --only-binary` will FAIL. No third-party arm64 wheel found.

## 2. Maintained macOS-arm64 fork — ONE REAL fork exists, but it targets py3.10 (NOT py3.12)
- Upstream google-research/football: real, last push 2025-06-17, not archived,
  90 open issues, 1365 forks. (confirmed 200)
- GitHub *repo* search `gfootball macos arm64` -> total_count 0 (no fork is
  discoverable by topic/description), BUT the upstream PR list surfaces a real one.
- REAL fork: **trygentic/football** (confirmed 200; is_fork=true, parent=
  google-research/football, default branch **`agentloop-build`**, pushed
  2026-05-27). It has ~30 branches incl. `master`, `upstream-pr-render-fixes`,
  several `try-*` build-attempt branches (`try-brew-sdl`, `try-sampler`,
  `try-profile`, `try-finish-pump`), and version tags v1.0–v2.9.
  https://github.com/trygentic/football
- It is the source of **PR #418** (open, by user `troyedwardsjr`, created/updated
  2026-05-27, head `trygentic:upstream-pr-render-fixes`) and **issue #417** (open).
  Both confirmed 200.
  - PR: https://github.com/google-research/football/pull/418
  - Issue: https://github.com/google-research/football/issues/417
- WHAT THE PR ACTUALLY IS (verified from the PR body): it fixes two macOS
  Apple-Silicon *rendering* bugs (sampler-incomplete texture binding; stuck
  Cocoa/NSOpenGL swap chain → black SDL window). The build story is a SIDE NOTE:
  the body says "My fork also includes build fixes for CMake 4 / Boost 1.85 /
  conda Python 3.10" and explicitly calls version-mismatch fixes "out of scope
  for this rendering PR."
- CRITICAL CAVEAT: per the PR author, the fork's build fixes target **conda
  Python 3.10 + Boost 1.85 + CMake 4**, NOT CPython 3.12 / Boost 1.90. No branch
  name advertises a 3.12 build. The interesting branch to read is the default
  `agentloop-build` (likely where the macOS build work lives) plus the `try-*`
  branches. So this is the closest thing to a maintained arm64 fork, but it does
  NOT solve py3.12 out of the box — worth reading `agentloop-build`'s CMake
  patches as a head-start.
- Other REAL upstream issues (each confirmed 200) documenting the same breakage,
  none with a merged arm64+py3.12 fix: #317 "Could NOT find Boost:" (open),
  #241 "libboost_python39", #363 "...Python 3.10 Conda environment", #342
  "Fail to install with conda environment on Mac", #192 "Conda support on MacOS".
- Best you can do: `pip install "git+https://github.com/trygentic/football.git"`
  WOULD pull the fork, but expect it to build only under the combo it targets
  (conda py3.10 + Boost 1.85), and the rendering fixes live on the
  `upstream-pr-render-fixes` branch, not master. For py3.12 you'd still rebuild
  Boost.Python (see #3). Treat as a starting point, not a turnkey fix.

## 3. Boost.Python for CPython 3.12 on Homebrew arm64 — WRONG PYTHON (it's 3.14)
- VERIFIED via Homebrew API: `boost-python3` is **1.90.0**, revision 0, and
  declares `"dependencies":["boost","python@3.14"]`. It is built/bottled against
  **python@3.14** — NOT 3.12 (and the user's "built for 3.14" assumption is
  CORRECT; my earlier 3.13 note was wrong). arm64 bottles exist:
  arm64_tahoe, arm64_sequoia, arm64_sonoma.
  (https://formulae.brew.sh/api/formula/boost-python3.json)
- Base `boost` formula (1.90.0) has ZERO python references -> ships no
  Boost.Python (confirms the user's report).
  (https://formulae.brew.sh/api/formula/boost.json)
- So `brew install boost-python3` yields `libboost_python314`, never
  `libboost_python312`. There is NO Homebrew route to a 3.12-linked
  Boost.Python.
- Least-effort 3.12 path = build Boost from source with a user-config jamfile
  pointing at python@3.12 (already installed at
  /opt/homebrew/bin/python3.12), e.g.:
    `./bootstrap.sh --with-python=/opt/homebrew/bin/python3.12`
    `./b2 --with-python python=3.12 cxxstd=17 stage`
  Rough cost on Apple Silicon: ~10-30 min for `--with-python` only (full Boost
  is longer), PLUS debugging gfootball's CMake to point at it and to drop the
  removed `boost_system` target (header-only since Boost 1.89).

## 4. Known-good version combo on macOS arm64 — PARTIAL (Boost 1.85 reported working)
- Multiple sources converge on **Boost 1.85** as the last version that builds
  with gfootball's CMake, because Boost **1.89** removed the compiled
  `libboost_system` that gfootball/configure scripts look for (it's header-only
  now). eos/eos issue #1078 documents the identical "configure fails with Boost
  1.89 due to removed libboost_system; downgrade to 1.85 works" pattern.
  (https://github.com/eos/eos/issues/1078)
- gfootball issue #317 "Could NOT find Boost:" (open) and #241
  "libboost_python39" document the Boost-version/python-version coupling on mac.
  (https://github.com/google-research/football/issues/317 ,
   https://github.com/google-research/football/issues/241)
- CAVEAT: the documented combo is Boost 1.85 + **Python 3.10**, NOT 3.12. No
  source verifies a clean Boost-1.7x/1.8x + gfootball + **py3.12** build on
  arm64. For py3.12 you must rebuild Boost.Python yourself (see #3).

## Ranked bottom line
There is NO turnkey path that works out of the box on arm64 + py3.12.
1. Lowest effort that actually runs: use a **linux-64 / x86_64 environment**
   (Docker or Rosetta/conda linux env). gfootball is Linux-first; this sidesteps
   Boost/CMake-on-mac entirely. Strictly out of the 4 hatches but the only
   reliable route. (Upstream ships gfootball/doc/docker.md.)
2. Native mac from scratch (only native route): `brew install boost-python3`
   gives a python@3.14-linked lib only, so build Boost.Python from source
   against /opt/homebrew/bin/python3.12, then patch gfootball's CMake
   (drop the removed boost_system, point Boost.Python at your build). Pin
   Boost 1.85 — Boost 1.89 removed libboost_system, which gfootball still
   references. High effort, unverified end-to-end.
3. There IS one real maintained fork (trygentic/football, PR #418) with macOS
   arm64 *rendering* fixes + build fixes — but for conda **py3.10 + Boost 1.85**,
   not py3.12. Useful as a base/reference, not a py3.12 turnkey. See #2.
4. conda package (any platform) and prebuilt arm64 wheel: DO NOT EXIST. conda
   returns 404 on conda-forge AND anaconda search is empty, so `conda install
   gfootball` fails on every platform, not just arm64.
