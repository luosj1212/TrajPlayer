TrajPlayer v0.1.0-alpha.9 distribution

Windows
1. Extract the entire TrajPlayer folder from the zip archive.
2. Run TrajPlayer.exe from inside that folder.
3. Keep the _internal folder next to TrajPlayer.exe.

Linux x86_64 (Ubuntu 22.04 or compatible)
1. Extract the entire TrajPlayer folder from the tar.gz archive.
2. Run ./TrajPlayer/TrajPlayer from a terminal, or double-click it in the file manager.
3. Keep the _internal folder next to TrajPlayer.
4. An OpenGL 3.3 capable graphics driver is required.

macOS 13 or newer
1. Choose the arm64 ZIP for Apple Silicon or the x86_64 ZIP for an Intel Mac.
2. Extract the entire TrajPlayer-macOS folder, then open TrajPlayer.app.
3. TrajPlayer.app can be moved to Applications as a whole.
4. The alpha app is not notarized. On first launch, Control-click the app, choose Open, and confirm Open.
5. An OpenGL 3.3 capable Mac is required.

The Windows/Linux executable cannot run if it is sent or moved without its _internal folder. On macOS, do not move files out of TrajPlayer.app.

Windows NumPy startup error
- Do not install Python or NumPy; they are already bundled.
- Make sure _internal\numpy\_core\_multiarray_umath.cp310-win_amd64.pyd exists.
- If it is missing, extract a fresh copy of the complete Release ZIP.
- Check Windows Security > Protection history if the file disappears after extraction.

Diagnostics
- Windows: TrajPlayer.exe --doctor-output=trajplayer-diagnostics.json
- Linux: ./TrajPlayer/TrajPlayer --doctor-output=trajplayer-diagnostics.json
- macOS: ./TrajPlayer.app/Contents/MacOS/TrajPlayer --doctor-output=trajplayer-diagnostics.json
- The report includes dependency and OpenGL driver information with local account paths redacted.

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
- Use the top-right controls button to show or hide the responsive inspector. Choose Chinese or English under Interface.
- Choose Ball-stick, Ball, or Bond from the representation control.
- Atom and bond sizes are independently adjustable with sliders. At 100%, Ball-stick uses 0.25x van der Waals atom radii and 0.20 A sticks; Ball uses full van der Waals radii.
- Use All, Chain, and Atom to control visibility. Chain accepts entries such as 1,3-5; Atom uses the adjacent slider.
- Bond status states when bonds were inferred from frame 1; clear Infer bonds to disable that static inference.
- Playback speed is adjustable from 1 to 60 FPS. Frames are always displayed in sequence and are never skipped.

Performance
- Traj, XTC, and TRR trajectories decode the requested directional window directly instead of creating a full float32 sidecar.
- RAM prefetch is selected automatically from frame size, available memory, renderer/topology reservations, read latency, and cache-hit behavior.
- Common XYZ and extXYZ rows are parsed natively into the frame cache. Frame 1 appears first while a reusable frame-offset index is built in the background.
- Chemfiles provides structure and XTC/TRR decoding; portable builds do not bundle SciPy or MDAnalysis.
- Hybrid-GPU Windows laptops automatically request the NVIDIA high-performance GPU when available.
- Keep current graphics drivers installed; TrajPlayer falls back to the available OpenGL 3.3 GPU.
