#!/usr/bin/env bash
# Creates a local .venv on first run, installs dependencies, then starts the API.
# GPU=1 ./run.sh  -> NVIDIA GPU build (much faster, ~2.5GB install); default is CPU.
set -e
cd "$(dirname "$0")"
MODE=cpu; REQ=api/requirements.txt
if [ "${GPU:-0}" = "1" ]; then MODE=gpu; REQ=api/requirements-gpu.txt; fi
if [ "$(cat .venv/.mode 2>/dev/null)" != "$MODE" ]; then
  rm -rf .venv
  python3 -m venv .venv
  .venv/bin/pip install -q -r "$REQ"
  echo "$MODE" > .venv/.mode
fi
exec .venv/bin/uvicorn api.main:app --host 0.0.0.0 --port "${PORT:-8000}"
