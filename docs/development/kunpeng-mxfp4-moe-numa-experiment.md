# Kunpeng 920 MXFP4 MoE CPU optimization - experiment plan

## Scope and baseline

- Repository: sharesdoc/llama.cpp
- Starting commit: `0f35d6d7b0f548993d4323eb1df8e7ca4cf039c5`
- Branch: `perf/kunpeng-mxfp4-numa`
- Target: CPU-only DeepSeek V4 Flash MXFP4 + DSpark on four Kunpeng 920 sockets (192 cores, 8 NUMA nodes).
- No production NPU, model weights, master branch, or existing CPU build modifications.

## Confirmed measurements supplied by the operator

- Single-request CPU decoding with 32 threads reaches about 1.89-2.04 generated tokens/s. Differences across prompt length and output length do not prove improvements.
- During decoding, 32 threads are active, aggregate CPU utilization about 2950%-3015%; four active threads per NUMA node.
- Memory pages associated with the process are highly skewed: N0=0.50, N1=24.09, N2=0.01, N3=6.77, N4=39.98, N5=7.27, N6=75.40, N7=6.24 GB (approximate 4-KiB-page equivalent). This does **not** establish remote-read bandwidth as the bottleneck.
- Historical llama-bench generation: threads 16=1.32, 32=1.76, 64=1.03 tokens/s. Measurements with different tools/conditions must not be directly mixed.

## Source audit at starting commit

1. `ggml/src/ggml-cpu/ggml-cpu.c`: `ggml_get_n_tasks()` assigns `n_threads` for both `GGML_OP_MUL_MAT` and `GGML_OP_MUL_MAT_ID`. Matrix outputs are partitioned across worker threads, so token-by-token decoding is not synonymous with single-thread execution.
2. `ggml_compute_forward_mul_mat_id()`: the MoE expert loop selects a chunking strategy. `disable_chunking = ggml_is_numa()` forces the simpler one-chunk-per-thread split under NUMA; otherwise the normal chunk size heuristic decides. A chunk counter distributes remaining work dynamically.
3. `ggml/src/ggml-cpu/ggml-cpu.c` traits: MXFP4 single-row dot uses `ggml_vec_dot_mxfp4_q8_0`, `nrows=1`.
4. `ggml/src/ggml-cpu/arch/arm/quants.c`: MXFP4 has NEON lookup + SDOT integer dot product; ARM compiled library contains SDOT.
5. `ggml/src/ggml-cpu/repack.cpp`: `mxfp4_4x4_q8_0` may be chosen when NEON and DotProd are available and the weight row shape is divisible by 4, but having a kernel does not prove it was dispatched for this workload.
6. Server `--parallel` / continuous batching supports multi-request throughput, a different goal from single-request decode latency.

## Hypothesis and first code experiment

In the NUMA case, forcing the per-thread partition of the `MUL_MAT_ID` output could perform worse than the existing dynamic chunk scheduling for some expert dimensions and active expert assignments. It could also perform **better** due to locality. We cannot infer the winner from aggregate CPU utilization alone.

Implement an opt-in, process-level **experimental** environment variable `GGML_CPU_MUL_MAT_ID_NUMA_CHUNKING=1` for the fallback `MUL_MAT_ID` path:

- Unset/default: preserve identical old behavior.
- Set to exactly `1`: retain the ordinary chunking heuristic even when `ggml_is_numa()` is true.
- Other values: preserve old behavior.
- Do not change math, model files, build flags, repack default, or non-NUMA scheduling. No dependency, model-specific CLI argument, or new API.
- Note: tiled / repack matmul paths may bypass this fallback entirely. If no A/B difference, verify dispatch before concluding that NUMA is irrelevant.

This is a controlled experiment, **not** a claimed throughput optimization. Do not merge without correctness and repeated performance data.

## Verification and acceptance criteria

1. Compile on a separate build directory on the AIS host; existing `build-cpu` and port 8002 must remain untouched. Run `test-backend-ops` for relevant CPU `MUL_MAT_ID` shapes, ideally with MXFP4.
2. On an isolated, nonproduction port, run baseline/default and opt-in versions with identical input model, threads, NUMA mode, DSpark settings, warmup, prompt, sampling, and max tokens. Compare median tokens/s and prefill latency over at least five runs, alternating A/B. Record acceptance rate and generated output, not just speed.
3. Test `--numa disabled` too: the environment variable must have no effect in that mode.
4. Check multi-request concurrent server behavior separately (`--parallel`, `--cont-batching`) to avoid conflating throughput with single-sequence latency.
5. Reject or revert if model output or backend operations regress, stability changes, or latency improvement is within noise. The operator independently verifies and accepts any result.

## Important dispatch finding

At this commit, `ggml/src/ggml-cpu/repack.cpp` implements its own `MUL_MAT_ID` `forward_mul_mat_id()` method for repacked CPU tensors. Its `supports_op()` can take a three-dimensional MXFP4 weight tensor when `CPU_REPACK` is the chosen buffer type and the activation input is F32. That path partitions output rows directly among threads; **it does not execute the fallback code changed in this branch**.

The experimental knob therefore probes only the *fallback* CPU MoE path. The benchmark matrix **must** compare `--repack` versus `--no-repack` before attributing any results to it. A null result with repack enabled is expected if every relevant MoE op is handled by the repack backend; it does not establish that chunking is unimportant.

Further source facts:
- `repack.cpp` loops selected experts in its own `forward_mul_mat_id()`. Each worker gets an output-row interval aligned to its `NB_COLS` grouping; this is parallel across workers but is a separate scheduler from the generic CPU backend.
- `tiled.cpp` also has a conditional `MUL_MAT_ID` dispatcher that may bypass the fallback when that dtype and expert batch size are eligible.
- A kernel existing in a compiled binary does not establish actual hot-path dispatch. Measure it.
- In particular, the observed 32 busy threads across eight NUMA nodes proves parallel CPU use, not efficient data locality or high SIMD utilization.

### Recommended independent A/B matrix

| Repack | NUMA | Environment override | Interpretation |
|---|---|---|---|
| on | distribute | unset | Current-like baseline |
| on | distribute | 1 | Should match baseline if repack handles the relevant MoE ops |
| off | distribute | unset | Fallback with conservative NUMA partitioning |
| off | distribute | 1 | Fallback with normal dynamic chunk heuristic |
| off | disabled | unset / 1 | Override should have no effect |

The isolated test process must not reuse port 8002 while the baseline service is up. Keep model weights read-only. Use fixed prompt and fixed token cap, warm up both variants, alternate runs, report medians and deviations. Do not interpret `--no-repack` as a production recommendation.

## Subsequent research gates

- Determine whether MXFP4 MoE fallback is actually used versus `repack` / tiled dispatch.
- Profile NUMA remote-memory traffic and hot loops; high CPU utilization is insufficient to identify a compute or memory bottleneck.
- Only after profiling, consider a dedicated MXFP4 ARM kernel change with bitwise/relative correctness tests across shapes, and benchmark against baseline.
