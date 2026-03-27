---
name: mlir-llvm-dump-for-perf-analysis
description: >
  Compare MLIR_ENABLE_DUMP=1 / LLVM_IR_ENABLE_DUMP=1 output files from two or more
  Triton compilations to find performance-related differences in pass ordering,
  layout encodings, vectorization, shared memory usage, and generated assembly.
  Use when diagnosing TFLOPS regressions, layout mismatches, or codegen differences
  between builds, plugin vs in-tree, or different pipeline configurations.
---

# Comparing Triton IR/LLVM Dump Files for Performance Differences

## Overview

Compare two or more `MLIR_ENABLE_DUMP=1 LLVM_IR_ENABLE_DUMP=1` output files from Triton compilations to identify performance-relevant differences in pass ordering, layout encodings, vectorization, shared memory usage, and generated assembly.

## Generating Dump Files

```bash
# Version A
MLIR_ENABLE_DUMP=1 LLVM_IR_ENABLE_DUMP=1 TRITON_ALWAYS_COMPILE=1 \
  python my_kernel.py &> /tmp/dump_a.txt

# Version B
MLIR_ENABLE_DUMP=1 LLVM_IR_ENABLE_DUMP=1 TRITON_ALWAYS_COMPILE=1 \
  python my_kernel.py &> /tmp/dump_b.txt
```

Key env vars:
- `MLIR_ENABLE_DUMP=1` — dumps IR before every MLIR pass (to stderr)
- `LLVM_IR_ENABLE_DUMP=1` — dumps LLVM IR during lowering (to stderr)
- `TRITON_ALWAYS_COMPILE=1` — bypasses cache, forces recompilation
- `pm.enable_debug()` — alternative: called in Python pipeline code, also dumps IR before each pass

Use `&>` (not `2>&1 |`) to capture both stdout and stderr to a file. Pipe-based capture may lose output or interleave incorrectly.

---

## Step 1: Compare Pass Ordering

Extract the pass sequence from each dump:

```bash
grep "^// -----// IR Dump Before" /tmp/dump_a.txt > /tmp/passes_a.txt
grep "^// -----// IR Dump Before" /tmp/dump_b.txt > /tmp/passes_b.txt
diff /tmp/passes_a.txt /tmp/passes_b.txt
```

What to look for:
- **Missing passes** — a pass present in one but not the other (e.g., layout propagation, async copy coalescing)
- **Reordered passes** — same passes but in different order (e.g., layout pass after vs before `RemoveLayoutConversions`)
- **Extra passes** — duplicate passes (e.g., `ConvertTritonToTritonGPU` running twice)
- **Pass name differences** — e.g., `TLXInsertRequireLayout` + `TlxPropagateLayout` (two passes) vs `TLXInsertAndPropagateLayout` (one combined pass)

Critical AMD pass ordering (layout pass must be between accelerate_matmul and remove_layout_conversions):
```
TritonAMDGPUAccelerateMatmul
<layout pass here>           <-- MUST be here
TritonGPURemoveLayoutConversions
TritonAMDGPUOptimizeEpilogue
```

---

## Step 2: Identify Compilation Boundaries

When autotuning, the dump contains multiple compilations (one per config). Identify them:

```bash
# Count and locate each compilation's module attributes
grep -n "num-warps" /tmp/dump_a.txt | sort | uniq -c | sort -rn
```

This shows which configs are compiled (e.g., 4-warp vs 8-warp). To compare a specific config, find its line range:

```bash
# Find key pass locations for each config
grep -n "IR Dump Before.*AccelerateMatmul\|IR Dump Before.*TLXInsert\|IR Dump Before.*RemoveLayoutConversions\|IR Dump Before.*ConvertTritonAMDGPUToLLVM" /tmp/dump_a.txt
```

The last occurrence of each pass typically corresponds to the largest/most interesting config.

---

## Step 3: Compare Shared Memory Encodings

Shared memory layout encoding directly affects load/store vectorization width.

```bash
# Extract shared encodings
grep "#shared\|swizzled_shared\|NVMMAShared" /tmp/dump_a.txt | sort -u
grep "#shared\|swizzled_shared\|NVMMAShared" /tmp/dump_b.txt | sort -u
```

Key attributes in `#ttg.swizzled_shared<{vec, perPhase, maxPhase, order}>`:
- **`vec`** — vectorization width. `vec=1` is unvectorized (bad). `vec=8` means 8-element vector loads from shared memory.
- **`perPhase`** and **`maxPhase`** — swizzle parameters. `perPhase=1, maxPhase=1` means no swizzling (default/basic). Higher values reduce bank conflicts.
- **`order`** — dimension ordering for memory access pattern.

Compare before vs after layout passes:
```bash
# Before layout pass (search near AccelerateMatmul)
sed -n '<LINE_BEFORE_LAYOUT>,<LINE_AFTER_LAYOUT>p' /tmp/dump_a.txt | grep "#shared"

# After layout pass (search near RemoveLayoutConversions)
sed -n '<LINE_AFTER_LAYOUT>,<LINE_NEXT_PASS>p' /tmp/dump_a.txt | grep "#shared"
```

Bad: `#ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, ...}>`
Good: `#ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 16, ...}>`

AMD must use `SwizzledSharedEncodingAttr` (not `NVMMASharedEncodingAttr`) for all SMEM shapes.

---

## Step 4: Compare Layout Encodings and Convert Layout Ops

Extra `convert_layout` ops mean unnecessary data movement:

```bash
# Count convert_layout ops in the inner loop at a given pass stage
sed -n '<START>,<END>p' /tmp/dump_a.txt | grep -c "convert_layout"
sed -n '<START>,<END>p' /tmp/dump_b.txt | grep -c "convert_layout"
```

Check `local_load` result types — they should match the DotOp's expected encoding:

```bash
# Good: local_load directly produces mma-compatible layout
ttg.local_load %src : !ttg.memdesc<...> -> tensor<...xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>

# Bad: local_load produces wrong layout, requiring convert_layout
ttg.local_load %src : !ttg.memdesc<...> -> tensor<...xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
%cvt = ttg.convert_layout %load : ... -> tensor<...xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
```

Extract blocked encoding parameters to check vectorization:
```bash
grep "#blocked" /tmp/dump_a.txt | sort -u
```
Key: `sizePerThread` — larger values mean better vectorization (e.g., `[1, 8]` > `[1, 1]`).

---

## Step 5: Compare Module Attributes

```bash
# Extract module attributes for each config
grep "module attributes" /tmp/dump_a.txt | sort -u
grep "module attributes" /tmp/dump_b.txt | sort -u
```

Key attributes:
- `ttg.num-warps` — warp count affects occupancy
- `ttg.shared` — total shared memory bytes allocated (appears after `AllocateSharedMemory` pass)
- `ttg.target` — must match (e.g., `hip:gfx950`)
- `ttg.num-ctas` — cluster size
- `tlx.has_explicit_local_mem_access` — metadata flag, does not affect AMD codegen

---

## Step 6: Compare Pipelining

Check `tt.num_stages` on `scf.for` loops:

```bash
# Find loop annotations near ScheduleLoops pass
sed -n '<SCHEDULE_LOOPS_LINE>,<PIPELINE_LINE>p' /tmp/dump_a.txt | grep "num_stages\|scf.for"
```

- `tt.num_stages = 0` means the kernel handles its own pipelining (explicit staging)
- `tt.num_stages = N` (N>1) means the compiler will software-pipeline with N stages

After the `Pipeline` pass, check for async copy ops:
```bash
grep -c "async_copy\|AsyncCopy\|buffer_load" /tmp/dump_a.txt
```

---

## Step 7: Compare Generated Assembly (AMDGCN)

If dumps include LLVM IR, or if `.amdgcn` files are in the cache:

```bash
# Find cached amdgcn files (after running with TRITON_CACHE_DIR)
find $TRITON_CACHE_DIR -name "*.amdgcn" | while read f; do echo "$(wc -l < "$f") $f"; done | sort -n

# Diff the largest configs
diff /tmp/cache_a/*/kernel.amdgcn /tmp/cache_b/*/kernel.amdgcn
```

What to look for in AMDGCN:
- **Instruction count** — fewer instructions = better (less register pressure, higher occupancy)
- **`v_mfma_*` instructions** — matrix multiply instructions; count should be the same
- **`ds_read_b128` vs `ds_read_b64` vs `ds_read_b32`** — wider shared memory reads are better
- **`buffer_load_dwordx4` vs `buffer_load_dword`** — wider global loads are better
- **`s_waitcnt`** — excessive wait counts indicate poor scheduling
- **`.vgpr_count` / `.sgpr_count`** — register pressure; lower is better for occupancy
- **`.lds_size`** — shared memory per workgroup

```bash
# Quick comparison of register usage and LDS
grep -E "vgpr_count|sgpr_count|lds_size|\.size" /tmp/cache_a/*/kernel.amdgcn
grep -E "vgpr_count|sgpr_count|lds_size|\.size" /tmp/cache_b/*/kernel.amdgcn
```

---

## Step 8: Compare Specific IR Sections

To diff the full IR at a specific pass stage between two dumps:

```bash
# 1. Find the line number of the pass in each dump
grep -n "IR Dump Before.*<PassName>" /tmp/dump_a.txt
grep -n "IR Dump Before.*<PassName>" /tmp/dump_b.txt

# 2. Extract the IR block (from that line to the next "IR Dump Before")
sed -n '<START>,<NEXT_PASS_LINE>p' /tmp/dump_a.txt > /tmp/ir_a.txt
sed -n '<START>,<NEXT_PASS_LINE>p' /tmp/dump_b.txt > /tmp/ir_b.txt

# 3. Diff ignoring location info and module attr metadata
diff /tmp/ir_a.txt /tmp/ir_b.txt | grep -v "^[<>].*#loc" | grep "^[<>]"
```

Best checkpoints to compare:
1. **After ConvertTritonToTritonGPU** — initial GPU layout assignment
2. **After AccelerateMatmul** — MMA instruction selection
3. **After layout pass** — shared memory encoding fixes
4. **After RemoveLayoutConversions** — final layout decisions
5. **After Pipeline** — software pipelining applied
6. **Before ConvertTritonAMDGPUToLLVM** — final TTGIR before lowering

---

## Step 9: Check for Runtime Overhead (Non-IR Issues)

If the IR and assembly are identical but performance differs, check runtime:

1. **Hook overhead** — `knobs.runtime.add_stages_inspection_hook` is called on EVERY kernel launch (not just compilation) to compute cache keys. If `get_key()` does file I/O (e.g., `pathlib.Path(__file__).read_text()`), cache the result:
   ```python
   _cached_key = None
   def get_key():
       global _cached_key
       if _cached_key is None:
           _cached_key = pathlib.Path(__file__).read_text()
       return _cached_key
   ```

2. **Autotuner contamination** — compilation overhead during `TRITON_ALWAYS_COMPILE=1` affects autotuner timing. Run with warm cache for accurate benchmarks.

3. **GPU thermal throttling** — run benchmarks individually, not back-to-back. Allow cooldown between runs.

4. **Cache key invalidation** — the hook hash changes if the source file changes, forcing recompilation. Use `TRITON_CACHE_DIR` to isolate caches.

---

## Quick Comparison Checklist

```bash
DUMP_A=/tmp/dump_a.txt
DUMP_B=/tmp/dump_b.txt

# 1. Pass ordering
diff <(grep "IR Dump Before" $DUMP_A) <(grep "IR Dump Before" $DUMP_B)

# 2. Shared encodings
diff <(grep "#shared\|swizzled_shared" $DUMP_A | sort -u) \
     <(grep "#shared\|swizzled_shared" $DUMP_B | sort -u)

# 3. Blocked encodings (vectorization)
diff <(grep "#blocked" $DUMP_A | sort -u) \
     <(grep "#blocked" $DUMP_B | sort -u)

# 4. Module attributes (warps, shared mem)
diff <(grep "module attributes" $DUMP_A | sort -u) \
     <(grep "module attributes" $DUMP_B | sort -u)

# 5. Convert layout count (fewer = better)
echo "A: $(grep -c 'convert_layout' $DUMP_A) convert_layouts"
echo "B: $(grep -c 'convert_layout' $DUMP_B) convert_layouts"

# 6. Async copy / pipelining ops
echo "A: $(grep -c 'async_copy\|AsyncCopy' $DUMP_A) async ops"
echo "B: $(grep -c 'async_copy\|AsyncCopy' $DUMP_B) async ops"

# 7. AMDGCN assembly sizes (if available)
for f in $CACHE_A/*/kernel.amdgcn; do wc -l "$f"; done | sort -n
for f in $CACHE_B/*/kernel.amdgcn; do wc -l "$f"; done | sort -n
```

---

## Common Performance-Impacting Differences

| Symptom | Likely Cause | Where to Look |
|---|---|---|
| Extra `convert_layout` in inner loop | Layout pass missing or misordered | After RemoveLayoutConversions |
| `vec=1` shared encoding | Layout pass not updating MemDesc | After layout pass, grep `#shared` |
| `NVMMAShared` on AMD | Wrong encoding selected for target | After ConvertTritonToTritonGPU |
| `sizePerThread=[1,1]` in inner loop | Poor blocked encoding for loads | After RemoveLayoutConversions |
| Different `ttg.shared` sizes | Different allocation due to layout | After AllocateSharedMemory |
| Missing `async_copy` ops | Pipelining not applied | After Pipeline pass |
| `ds_read_b32` instead of `ds_read_b128` | Unvectorized shared mem access | AMDGCN assembly |
| Identical IR but different perf | Runtime overhead or autotuner noise | Hook overhead, cache, GPU thermals |
