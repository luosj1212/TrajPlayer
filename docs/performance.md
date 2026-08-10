# Performance Validation

TrajPlayer keeps performance checks separate from visual defaults. The normal
renderer, ball-and-stick appearance, controls, and sequential no-skip playback
are used by the synthetic benchmark.

## Reference Benchmark

Run the 100,000 atom and 99,999 bond reference scene:

```bash
python app.py --benchmark-output=benchmark-current.json \
  --benchmark-atoms=100000 --benchmark-frames=64 \
  --benchmark-render-frames=120 --benchmark-bonds \
  --benchmark-finish-gpu
```

Validate the report schema and correctness invariants:

```bash
python scripts/check_perf_report.py benchmark-current.json
```

The validator fails when playback drops or duplicates a trajectory frame, a
stale lease is released, the renderer copies a complete CPU frame, the run
times out, or a required metric is absent.

## Relative Regression Check

Compare results recorded on the same machine, display configuration, and GPU
driver:

```bash
python scripts/compare_perf.py benchmark-baseline.json \
  benchmark-current.json --fail-regression-percent=10
```

The comparison covers startup/open latency; render, upload, present, and frame
read p50/p95/p99; cadence; decode throughput; cache hit rate; and process RSS.
Absolute timings vary by GPU and compositor, so same-machine relative results
are the primary signal.

## Real-File Scenarios

The synthetic scene isolates renderer performance. Use the real-file runner to
measure the reader, progressive index, seek, and streaming-cache paths with a
representative local trajectory:

```bash
python scripts/bench_scenarios.py open --trajectory run.xtc \
  --topology run.gro --repeat 5 --output perf/xtc-open.json

python scripts/bench_scenarios.py seek --trajectory run.xtc \
  --topology run.gro --samples 500 --pattern random \
  --output perf/xtc-seek.json

python scripts/bench_scenarios.py soak --trajectory run.xtc \
  --topology run.gro --minutes 30 --fps 60 \
  --output perf/xtc-soak.json
```

`open` records metadata, first-frame, and optional progressive-index latency.
`seek` supports random, sequential, and alternating seek-storm patterns.
`soak` consumes every frame in order, records deadline misses without skipping,
and reports RSS slope, cache behavior, and lease invariants. OS file-cache state
is recorded as unmanaged; cold and warm runs should therefore be labeled and
collected separately by the operator.

Compare reports for the same scenario with the existing relative tool:

```bash
python scripts/compare_perf.py perf/xtc-seek-baseline.json \
  perf/xtc-seek-current.json --fail-regression-percent=10
```

The comparison warns when machine, processor, or platform metadata differs.

Benchmark the native million-atom depth-order path independently with:

```bash
python scripts/bench_depth_order.py --atoms 1000000 --repeat 9 --require-native
```

Measure common XYZ/extXYZ row parsing and greedy valence selection without the
UI or renderer:

```bash
python scripts/bench_xyz_parser.py --atoms 1000000 --repeat 3
python scripts/bench_valence_selection.py --atoms 200000 \
  --candidates 1000000 --repeats 3
```

To expose the deferred million-atom camera-stop work, add
`--benchmark-camera-spin-seconds=5` to the synthetic benchmark. The report then
separates depth ordering, CPU array rebuild, static permutation upload, and the
first frame after interaction stops. Zero-work frames are not mixed into these
event-only latency distributions.

## Required JSON Groups

- `startup`: process to QApplication, visible window, and first GL frame
- `open`: metadata, first useful frame, and progressive index completion
- `render`: cadence plus paint, upload, and depth-sort latency distributions
- `pipeline`: present latency, dropped frames, and duplicate frames
- `io`: read latency, throughput, hit rate, prefetch, slab, and lease counters
- `memory`: idle/playback/peak RSS and allocated frame-cache bytes
- `copies`: actual renderer full-frame fallback copy bytes

Latency samples use fixed-size NumPy rings, so telemetry memory does not grow
with playback length.

## Package Forensics

After building a portable package, generate a size report and enforce the
checked-in platform baseline:

```bash
python scripts/report_bundle_size.py \
  --json-output=bundle-size.json \
  --baseline-file=scripts/bundle_size_baselines.json \
  --platform=Windows-x64 --max-growth-percent=2
```

Use `Linux-x86_64`, `macOS-arm64`, or `macOS-x86_64` for the other release
targets. The report includes group totals, largest files and dynamic libraries,
SHA-256 duplicate sets, and baseline deltas.

`scripts/bench_linux_strip.py` can A/B a Linux tarball without modifying it.
It strips a copied runtime tree, runs startup/native/reader smoke tests, and
reports both tree and compressed-archive deltas. NumPy wheel libraries and the
PyInstaller launcher are intentionally excluded because post-build GNU strip
can invalidate their ELF layout or embedded package section.

## Release Coverage

Pull-request CI runs the full test suite and reader smoke corpus on Windows,
Linux, and macOS. Linux additionally records a synthetic performance artifact;
Windows builds and checks the complete portable folder. GitHub's hosted Windows
desktop does not expose the required OpenGL 3.3 context, so tagged releases run
the 100,000 atom benchmark on Linux and both macOS targets and retain each JSON
report with the package artifact. Windows GPU results should be recorded on a
real OpenGL 3.3-capable machine using the reference command above.
