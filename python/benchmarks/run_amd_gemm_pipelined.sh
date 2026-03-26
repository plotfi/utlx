#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRITON_TLX_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
# When built as a submodule of triton, the .so lives in the triton build output.
# Set TRITON_REPO_ROOT to override if triton is not the immediate parent.
TRITON_REPO_ROOT="${TRITON_REPO_ROOT:-$(cd "$TRITON_TLX_ROOT/.." && pwd)}"

TRITON_PASS_PLUGIN_PATH="$TRITON_REPO_ROOT/python/triton/plugins/libTLXMemOpsPlugin.so" \
    python "$SCRIPT_DIR/amd-gemm-pipelined.py" "$@"
