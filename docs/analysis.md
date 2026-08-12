# Scientific Analysis

TrajPlayer's analysis pane is intended for fast inspection of the trajectory
already open in the viewer. It uses NumPy implementations and the same
frame/cell data shown by the viewport. It is not a replacement for a fully
scripted, reproducible analysis workflow.

## Time And Frame Ranges

The default horizontal coordinate is the zero-based frame number. TrajPlayer
uses physical time only when a positive frame interval is supplied in fs, ps,
or ns. Analysis uses the full trajectory unless Playback range is enabled; the
analysis stride is then applied to that range.

Only one trajectory scan runs at once. It owns one reusable full-frame float32
slab and copies only the selected coordinates needed by an algorithm. During
playback and live timeline scrubbing, uncached analysis reads yield so rendering
and streaming retain priority. Closing or replacing a trajectory cancels its
analysis generation.

## Density

Number density is `N / V` in `1/A3`. Mass density uses atomic masses from ASE
and converts `amu/A3` to `g/cm3`. A valid cell is required for every sampled
frame. Bulk density and density profiles always use every atom in the system;
an active viewport selection does not change their scope.

Density profiles map positions to fractional cell coordinates before binning
along X, Y, or Z. This keeps periodic binning meaningful for triclinic cells.
Each profile is normalized by its slice volume. Heatmaps retain at most 2,000
time rows for bounded display and result size.

## Mean-Squared Displacement

MSD from the origin reports the selected-atom mean of
`|r(t) - r(0)|^2`. Time-averaged MSD averages the same quantity over every
available time origin for each lag. X, Y, Z, XY, and XYZ components are
available.

For periodic trajectories, enable PBC/no-jump. Consecutive Cartesian positions
are converted to fractional coordinates, the fractional displacement is
wrapped to the nearest image, and that displacement is transformed by the
current frame's cell. Every sampled frame must have a valid cell. The optional
COM-drift control subtracts the selected atoms' translational drift.

Time-averaged MSD writes selected float32 coordinates to a temporary disk-backed
array. Small jobs use a direct calculation; larger jobs use atom-chunked NumPy
FFT autocorrelation. Max lag limits both output and work. Temporary data is
removed when the job succeeds, fails, or is cancelled.

TrajPlayer intentionally does not calculate a final diffusion coefficient.
Short-time ballistic motion, long-time sampling noise, finite-size effects, and
the choice of linear fit interval require scientific judgment.

## RMSD And RMSF

RMSD compares each sampled frame with the selected reference frame. Align/fit
removes translation and applies a proper Kabsch rotation; Mass weighted uses
ASE atomic masses. RMSF uses streaming Welford accumulation and reports one
value per selected atom, so all frames are never retained in memory.

PBC/no-jump first makes the selection whole relative to its first atom. This is
appropriate for one connected molecule but can be ambiguous for disconnected
groups or a complete solvent box.

## Center Of Mass And Radius Of Gyration

COM and Rg use ASE atomic masses. With PBC enabled, selected coordinates are
made whole relative to an anchor before evaluation. Unknown or invalid element
identities disable mass-dependent analysis instead of silently assigning a
mass.

## Measurements

Two selected atoms define a distance, three define an angle, and four define a
signed dihedral. The same geometry functions are used for the current viewport
annotation and for a pinned measurement's time series. Periodic mode uses a
fractional-coordinate minimum image and therefore supports triclinic cells.

## References

- [MDAnalysis mean-squared displacement documentation](https://docs.mdanalysis.org/stable/documentation_pages/analysis/msd.html)
  discusses no-jump coordinates, windowed MSD, FFT acceleration, and diffusion
  fit limitations.
- [GROMACS periodic boundary conditions](https://manual.gromacs.org/current/reference-manual/algorithms/periodic-boundary-conditions.html)
  describes rectangular and triclinic simulation cells.
- Wolfgang Kabsch, [A solution for the best rotation to relate two sets of vectors](https://doi.org/10.1107/S0567739476001873),
  Acta Crystallographica Section A 32, 922-923 (1976).
- [NumPy linear algebra documentation](https://numpy.org/doc/stable/reference/routines.linalg.html)
  covers the SVD and matrix operations used by the alignment implementation.

For publication or production analysis, record all selections, frame ranges,
strides, PBC treatment, reference frames, units, and software versions, then
cross-check important values with a scripted tool or the simulation package.
