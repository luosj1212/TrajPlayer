# Changelog

All notable changes to TrajPlayer are documented here. The project follows
[Semantic Versioning](https://semver.org/) once it reaches a stable release.

## [Unreleased]

## [0.1.0-alpha.11] - 2026-08-12

### Changed

- Density-over-time and density-profile analyses now always use the entire
  system, independent of the current atom selection
- Split the responsive Inspector into View, Inspect, and Analysis tabs so all
  analysis parameters and actions remain visible at supported window sizes
- Clarified whole-system and current-selection analysis scope in both Chinese
  and English interfaces

## [0.1.0-alpha.10] - 2026-08-12

### Added

- Added GPU atom picking with integer-ID, RGBA8, and vectorized CPU fallbacks;
  canonical selections persist across frames and render through a compact GPU
  selection texture
- Added triclinic minimum-image distance, angle, and dihedral measurements,
  pinned viewport annotations, and measurement-over-time analysis
- Added a marker- and range-aware timeline linked bidirectionally to analysis
  curves while retaining live coalesced scrubbing and sequential no-skip playback
- Added cancellable background density, density-profile, MSD, RMSD, RMSF, COM,
  and radius-of-gyration analysis with bounded frame slabs and disk-backed
  windowed MSD storage
- Added a dependency-free line/heatmap plot with min/max decimation, HiDPI data
  caching, linear/log axes, zoom, tooltips, CSV/PNG export, and exact frame seek
- Added recent trajectory sources, richer drag-and-drop feedback, current-frame
  XYZ/extXYZ export, viewport screenshots, Light/Dark/System themes, and Qt
  `.qm` Chinese/English translations

### Changed

- Expanded the responsive Inspector around Display, Visibility, Selection,
  Measurement, Timeline, Analysis, Playback, and Interface workflows
- Kept analysis I/O serialized and lower priority than playback or live
  scrubbing; inactive analysis and picking perform no periodic work
- Preserved the alpha.9 renderer, molecular shading, ball-and-stick proportions,
  antialiasing, frame-streaming cache, and no-skip playback path

### Performance

- A local RTX 4070 Laptop GPU run sustained 60.00 FPS for 100,000 atoms and
  99,999 bonds with 2.83 ms paint p95 and 0.39 ms position-upload p95
- The local one-million-atom camera-stop frame measured 23.02 ms, and GPU
  picking measured 5.37 ms p95; results depend on hardware and driver

## [0.1.0-alpha.9] - 2026-08-10

### Added

- Added a responsive right-side inspector, compact transport bar, collapsible
  advanced controls, and runtime Chinese/English interface switching
- Added a five-second camera spin/stop benchmark with separate depth-order,
  array-rebuild, static-upload, and first-post-interaction timing
- Added standalone million-atom XYZ parser and million-candidate valence
  selection benchmarks

### Changed

- Moved common XYZ/extXYZ atom-row parsing into native `trajcore`, writing
  directly into the streamer's contiguous float32 slab while retaining the
  Python fallback for unusual `Properties` layouts
- Reused the metadata read of frame 1 for initial display in ASE, XYZ/extXYZ,
  XTC, and TRR readers instead of decoding it twice
- Kept canonical atom color, radius, and periodic-anchor attributes on the GPU;
  deferred camera sorting now rebuilds and uploads only an atom permutation
- Moved bond-candidate sorting and valence-cap selection into native code while
  preserving the existing shortest-first greedy result
- Strengthened packaged native smoke tests to require every alpha.9 hot path

### Fixed

- Closed XYZ source mappings when progressive-index construction fails
- Included the macOS build helper in WSL's isolated Linux release workspace

## [0.1.0-alpha.8] - 2026-08-10

### Changed

- Replaced the million-atom depth-bin `stable argsort` with a native
  `O(N + 256)` counting pass while preserving the previous far-to-near order
- Added paint-confirmed `RenderTicket` ownership so unrelated OpenGL swaps
  cannot acknowledge a pending trajectory frame
- Made adaptive cache decisions use the selected playback FPS, decode deadline,
  and a bounded process-RSS soft limit instead of assuming 60 FPS
- Added real-file open, seek-storm, and no-skip streaming-soak benchmarks for
  ASE, XYZ/extXYZ, and Gromacs trajectories
- Extended relative performance comparison to understand real-file scenario
  reports and warn when hardware or platform context differs

## [0.1.0-alpha.7] - 2026-08-09

### Changed

- Replaced the window-owned present queue state with a generation-safe
  `PresentScheduler` while retaining sequential, no-skip playback
- Made frame leases read-only and epoch checked, deprecated unleased frame
  getters, and added acquisition/release/stale-reference diagnostics
- Added foreground-priority XYZ indexing I/O, resumable atomic checkpoints, and
  throttled background scanning
- Changed the RAM frame cache to dynamically allocated slabs with pressure-aware
  shrink/grow hysteresis and Linux cgroup awareness
- Added bounded p50/p95/p99 telemetry for render, upload, present, depth-sort,
  frame-read, process-memory, and renderer-copy behavior
- Added a 256-bin coarse depth order for million-atom scenes to bound post-input
  sorting latency without changing atom or bond appearance
- Added performance report validation/comparison tools and performance artifacts
  to cross-platform release builds
- Added per-library and per-file bundle forensics, duplicate binary detection,
  Qt module/plugin allowlists, and a two-percent unexplained package growth gate
- Deferred nonessential backend, diagnostics, worker, and benchmark imports to
  improve cold startup without changing the visible UI
- Made OpenGL teardown and not-yet-ready position buffers safe during pending
  paint events, and stopped release CI from forcing an OpenGL 3.0 Windows
  software renderer against TrajPlayer's documented OpenGL 3.3 requirement

## [0.1.0-alpha.6] - 2026-08-09

### Changed

- Added leased frame-cache slots so OpenGL uploads use the streamer's contiguous
  frame directly without a second renderer-side full-frame copy
- Replaced synchronous playback repainting with a one-frame present queue that
  advances only after Qt emits `frameSwapped`; sequential playback still never
  skips trajectory frames
- Deferred camera depth sorting until pointer interaction is idle, avoiding
  repeated `O(N log N)` sorts and static VBO uploads while dragging
- Replaced the fixed 256 MiB RAM cache with a 64-256 MiB adaptive budget that
  accounts for available memory, frame size, renderer/topology working sets,
  optional writers, decode latency, and cache-hit telemetry
- Changed ASE `.traj`, XYZ/extXYZ, XTC, and TRR inputs to direct readers; a full
  decoded sidecar is no longer required or populated in the background
- Added progressive XYZ/extXYZ indexing so the first frame is available before
  the complete file-offset scan finishes
- Replaced the MDAnalysis Gromacs backend with Chemfiles and removed SciPy from
  bond inference by adding union-find and cell-list implementations
- Added the optional native `trajcore` extension for neighbor candidates and
  connected components, with a tested NumPy fallback
- Added lightweight direct ASE ULM, extXYZ, GRO, PDB, and CIF read paths so
  portable builds never fall through an `ase.io` import that requires SciPy
- Added source and packaged all-format reader smoke tests for `.traj`, extXYZ,
  GRO, PDB, CIF, XTC, and TRR
- Split Qt view construction, background workers, and QAction commands out of
  the application controller without changing the visible layout
- Added dependency-level portable bundle size reporting to release builds

## [0.1.0-alpha.5] - 2026-08-08

### Added

- Explicit bond-topology source state, a visible frame-1 inference label, and
  an option to disable inferred bonds
- A privacy-safe diagnostics command that reports dependency and OpenGL driver
  details
- Real two-frame GUI/OpenGL smoke tests for source and packaged Windows/Linux
  builds
- Native macOS 13+ app bundles for Apple Silicon and Intel, including Finder
  file-open handling and architecture-specific Release ZIPs

### Changed

- Sidecar metadata, shapes, member paths, and exact buffer byte sizes are
  validated before any memory map is created
- The render timer is event-driven and sleeps while the viewer is idle
- Error logs rotate at 2 MiB and redact local account paths from startup data
- Chain isolation accepts typed lists and ranges such as `1,3-5`

### Fixed

- Frame-streaming failures now propagate to the UI, and mapped stores remain
  owned until background readers have actually stopped
- Shutdown waits use a shared deadline instead of unbounded worker waits
- The Open action now uses the platform-standard Ctrl+O/Cmd+O shortcut
- Periodic chain isolation now follows each chain's current-frame GPU anchor
  instead of pinning one atom to the coordinate where the filter was applied

## [0.1.0-alpha.4] - 2026-08-07

### Fixed

- Reassembled isolated chains across periodic boundaries with a stable,
  center-near anchor computed when the filter changes
- Applied minimum-image bond vectors in the GPU shader so wrapped coordinates
  no longer produce box-spanning sticks
- Included periodic neighbors during background bond inference
- Kept per-frame rendering on the existing float32 VBO upload path without
  adding per-atom CPU geometry work

## [0.1.0-alpha.3] - 2026-08-07

### Fixed

- Configured bundled NumPy and SciPy DLL directories before importing NumPy
- Replaced the misleading NumPy source-tree traceback with an actionable damaged-package message
- Disabled optional UPX packing to reduce interference from endpoint security software
- Added native runtime-file validation and packaged executable smoke tests to Release CI

## [0.1.0-alpha.2] - 2026-08-01

### Fixed

- Prevented rapid consecutive opens from rebuilding the same cache concurrently
- Added an isolated session-cache fallback when Windows keeps a memmap cache locked
- Made Windows Conda packaging collect required runtime DLLs without an activated shell

## [0.1.0-alpha.1] - 2026-07-30

### Added

- OpenGL 3.3 instanced sphere and bond rendering
- Float32 memmap sidecar trajectory store with bounded streaming cache
- Background target-first loading and live slider scrubbing
- ASE trajectory, XYZ/extXYZ, PDB, CIF, GRO, XTC, and TRR input paths
- Ball-stick, Ball, and Bond representations with independent size controls
- Periodic box display and chain/atom isolation controls
- Adjustable 1-60 FPS sequential playback without trajectory-frame skipping
- Windows x64 and Linux x86_64 portable packaging
- Cross-platform tests, CI, release automation, and dependency license collection

[Unreleased]: https://github.com/luosj1212/TrajPlayer/compare/v0.1.0-alpha.11...HEAD
[0.1.0-alpha.11]: https://github.com/luosj1212/TrajPlayer/releases/tag/v0.1.0-alpha.11
[0.1.0-alpha.10]: https://github.com/luosj1212/TrajPlayer/releases/tag/v0.1.0-alpha.10
[0.1.0-alpha.9]: https://github.com/luosj1212/TrajPlayer/releases/tag/v0.1.0-alpha.9
[0.1.0-alpha.8]: https://github.com/luosj1212/TrajPlayer/releases/tag/v0.1.0-alpha.8
[0.1.0-alpha.7]: https://github.com/luosj1212/TrajPlayer/releases/tag/v0.1.0-alpha.7
[0.1.0-alpha.6]: https://github.com/luosj1212/TrajPlayer/releases/tag/v0.1.0-alpha.6
[0.1.0-alpha.5]: https://github.com/luosj1212/TrajPlayer/releases/tag/v0.1.0-alpha.5
[0.1.0-alpha.4]: https://github.com/luosj1212/TrajPlayer/releases/tag/v0.1.0-alpha.4
[0.1.0-alpha.3]: https://github.com/luosj1212/TrajPlayer/releases/tag/v0.1.0-alpha.3
[0.1.0-alpha.2]: https://github.com/luosj1212/TrajPlayer/releases/tag/v0.1.0-alpha.2
[0.1.0-alpha.1]: https://github.com/luosj1212/TrajPlayer/releases/tag/v0.1.0-alpha.1
