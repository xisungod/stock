#!/bin/sh
set -eu

DEPLOY_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PYTHON=${PYTHON:-python3}
WHEELS="$DEPLOY_DIR/wheels"

# jsonpath is pure Python but PyPI only distributes its source archive.
"$PYTHON" -m pip wheel --no-deps --wheel-dir "$WHEELS" jsonpath==0.82.2
"$PYTHON" -m pip download \
    --no-deps \
    --dest "$WHEELS" \
    --find-links "$WHEELS" \
    --only-binary=:all: \
    --platform manylinux2014_x86_64 \
    --platform manylinux_2_27_x86_64 \
    --platform manylinux_2_28_x86_64 \
    --python-version 3.12 \
    --implementation cp \
    --abi cp312 \
    -r "$DEPLOY_DIR/requirements.lock"
