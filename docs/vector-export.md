# Vector Export

TrajPlayer can export only the molecular image currently visible in the 3D
viewport as an editable SVG file. Open **View > Advanced** and choose
**Export vector SVG**.

The export freezes the displayed scene at the moment the command is used. It
preserves:

- the current camera rotation, pan, zoom, projection, and viewport crop
- Ball-stick, Ball, or Bond representation and the current atom/bond sizes
- the current chain/atom visibility filter and atom selection highlighting
- atom and two-tone bond colors
- the current background and the periodic box when it is visible

The SVG contains native ellipses, bond groups, gradients, and box lines. It
does not embed a PNG and does not include TrajPlayer controls, the Inspector,
timeline, measurement labels, or other application overlays. Atom elements
also retain their zero-based canonical index in `data-atom-index`.

SVG is a 2D projection of the current view, not a saved interactive 3D scene.
Its vector lighting closely follows the OpenGL ball-and-stick appearance, but
GPU pixel shading is represented with editable SVG gradients rather than
captured pixel for pixel. Use the existing **Save screenshot** command when an
exact raster copy of every rendered pixel is required.

Large molecular systems create correspondingly large SVG files because every
visible atom and bond remains individually editable. Encoding and file I/O run
in a background worker, and cancelling or closing TrajPlayer removes any
incomplete temporary file.
