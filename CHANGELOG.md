# Changelog

All notable changes to TrajPlayer are documented here. The project follows
[Semantic Versioning](https://semver.org/) once it reaches a stable release.

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

[0.1.0-alpha.5]: https://github.com/luosj1212/TrajPlayer/releases/tag/v0.1.0-alpha.5
[0.1.0-alpha.4]: https://github.com/luosj1212/TrajPlayer/releases/tag/v0.1.0-alpha.4
[0.1.0-alpha.3]: https://github.com/luosj1212/TrajPlayer/releases/tag/v0.1.0-alpha.3
[0.1.0-alpha.2]: https://github.com/luosj1212/TrajPlayer/releases/tag/v0.1.0-alpha.2
[0.1.0-alpha.1]: https://github.com/luosj1212/TrajPlayer/releases/tag/v0.1.0-alpha.1
