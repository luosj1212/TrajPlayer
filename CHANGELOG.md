# Changelog

All notable changes to TrajPlayer are documented here. The project follows
[Semantic Versioning](https://semver.org/) once it reaches a stable release.

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

[0.1.0-alpha.4]: https://github.com/luosj1212/TrajPlayer/releases/tag/v0.1.0-alpha.4
[0.1.0-alpha.3]: https://github.com/luosj1212/TrajPlayer/releases/tag/v0.1.0-alpha.3
[0.1.0-alpha.2]: https://github.com/luosj1212/TrajPlayer/releases/tag/v0.1.0-alpha.2
[0.1.0-alpha.1]: https://github.com/luosj1212/TrajPlayer/releases/tag/v0.1.0-alpha.1
