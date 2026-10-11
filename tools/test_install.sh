#!/bin/sh
set -eu

repo=$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd)
venv=$(mktemp -d)
trap 'rm -rf "$venv"' EXIT HUP INT TERM

"${PYTHON:-python3}" -m venv "$venv"
"$venv/bin/python" -m pip install --disable-pip-version-check --timeout 30 --retries 1 \
    -r "$repo/sim/requirements.txt"
PATH="$venv/bin:$PATH" make -C "$repo" test
