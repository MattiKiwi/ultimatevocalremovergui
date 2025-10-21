#!/usr/bin/env bash
#
# Bootstrap a UV-managed virtual environment for Ultimate Vocal Remover.
# Requires the `uv` CLI (https://github.com/astral-sh/uv) to be installed.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${REPO_ROOT}/.venv"

if ! command -v uv >/dev/null 2>&1; then
    echo "error: uv CLI not found in PATH. Install it via https://github.com/astral-sh/uv" >&2
    exit 1
fi

echo "Creating UV virtual environment at ${VENV_DIR}"
uv venv "${VENV_DIR}" --python 3.10

echo "Synchronising dependencies from pyproject.toml"
uv sync --python "${VENV_DIR}/bin/python"

cat <<'EOT'

UV environment ready.
Activate it with:

  source .venv/bin/activate

Or run commands without activating:

  uv run --python .venv/bin/python <command>

EOT
