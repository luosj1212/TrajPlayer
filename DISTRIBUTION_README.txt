TrajPlayer v0.1.0-alpha.4 distribution

Windows
1. Extract the entire TrajPlayer folder from the zip archive.
2. Run TrajPlayer.exe from inside that folder.
3. Keep the _internal folder next to TrajPlayer.exe.

Linux x86_64 (Ubuntu 22.04 or compatible)
1. Extract the entire TrajPlayer folder from the tar.gz archive.
2. Run ./TrajPlayer/TrajPlayer from a terminal, or double-click it in the file manager.
3. Keep the _internal folder next to TrajPlayer.
4. An OpenGL 3.3 capable graphics driver is required.

The executable cannot run if it is sent or moved without its _internal folder.

Windows NumPy startup error
- Do not install Python or NumPy; they are already bundled.
- Make sure _internal\numpy\_core\_multiarray_umath.cp310-win_amd64.pyd exists.
- If it is missing, extract a fresh copy of the complete Release ZIP.
- Check Windows Security > Protection history if the file disappears after extraction.

Project and issue tracker
https://github.com/luosj1212/TrajPlayer

License
- TrajPlayer source is provided under the MIT License.
- See LICENSE, THIRD_PARTY_NOTICES.md, and the licenses directory included in
  this package for dependency license terms.

Gromacs trajectories
- Open a GRO structure by itself, or select/drop one GRO topology together with one XTC or TRR trajectory.
- When an XTC/TRR file has a same-named GRO file beside it, opening the trajectory alone also works.

Display
- Choose Ball-stick, Ball, or Bond from the representation control.
- Atom and bond sizes are independently adjustable with sliders. At 100%, Ball-stick uses 0.25x van der Waals atom radii and 0.20 A sticks; Ball uses full van der Waals radii.
- Use the All, Chain, and Atom segments plus the adjacent slider to control which atoms are visible.
- Playback speed is adjustable from 1 to 60 FPS. Frames are always displayed in sequence and are never skipped.

Performance
- Traj, XTC, and TRR trajectories load requested frames directly while the float32 cache fills in the background.
- XYZ and extXYZ trajectories use a reusable frame-offset index for direct frame reads.
- Hybrid-GPU Windows laptops automatically request the NVIDIA high-performance GPU when available.
- Keep current graphics drivers installed; TrajPlayer falls back to the available OpenGL 3.3 GPU.
