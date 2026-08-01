# Changelog

All notable changes to TrajPlayer are documented here. The project follows
[Semantic Versioning](https://semver.org/) once it reaches a stable release.

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

[0.1.0-alpha.2]: https://github.com/luosj1212/TrajPlayer/releases/tag/v0.1.0-alpha.2
[0.1.0-alpha.1]: https://github.com/luosj1212/TrajPlayer/releases/tag/v0.1.0-alpha.1
