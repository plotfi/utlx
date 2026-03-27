# Standalone build for µTLX

The CMakeLists.txt is now a top-level CMake project requiring three paths:

```
  ┌───────────────────┬──────────────────────────────┐
  │     Variable      │         Description          │
  ├───────────────────┼──────────────────────────────┤
  │ TRITON_SOURCE_DIR │ Triton source tree root      │
  ├───────────────────┼──────────────────────────────┤
  │ TRITON_BUILD_DIR  │ Triton CMake build directory │
  ├───────────────────┼──────────────────────────────┤
  │ LLVM_BUILD_DIR    │ LLVM/MLIR build directory    │
  └───────────────────┴──────────────────────────────┘
```

These can be passed via -D flags or environment variables. Usage:

```
  # Be Sure to Build your Triton with `export TRITON_EXT_ENABLED=1`
  cmake -B build \
    -DTRITON_SOURCE_DIR=/path/to/triton \
    -DTRITON_BUILD_DIR=/path/to/triton/build/cmake.linux-x86_64-cpython-3.11 \
    -DLLVM_BUILD_DIR=/path/to/triton/llvm-project/build
  ninja -C build
```

The output is build/libTLXMemOpsPlugin.so, which can be loaded via:
TRITON_PLUGIN_PATHS=utlx/build/libTLXMemOpsPlugin.so python my_kernel.py

