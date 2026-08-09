# Third-Party Notices

TrajPlayer is MIT-licensed software built with the following third-party
projects. Each project remains subject to its own license.

| Component | Version used by current builds | License |
| --- | ---: | --- |
| [Atomic Simulation Environment](https://ase-lib.org/) | 3.27.0 | LGPL-2.1-or-later |
| [Chemfiles](https://chemfiles.org/) | 0.10.4 | BSD-3-Clause |
| [NumPy](https://numpy.org/) | 2.2.6 | BSD-3-Clause |
| [PySide6 / Qt for Python](https://doc.qt.io/qtforpython-6/) | 6.10.2 | LGPL-3.0-only OR GPL-2.0-only OR GPL-3.0-only; distributed here under LGPL-3.0 |
| [PyInstaller](https://pyinstaller.org/) | 6.19.0 | GPL-2.0-or-later with the PyInstaller bootloader exception |

Release archives contain a `licenses` directory populated from the installed
package metadata at build time. That directory contains the license texts
shipped by the corresponding dependency distributions.

TrajPlayer does not modify Qt, PySide6, ASE, or Chemfiles. The onedir release
layout keeps dependency libraries as separate files in `_internal`; users may
replace compatible library files in accordance with the applicable licenses.

The names of third-party projects and their contributors may not be used to
endorse TrajPlayer without permission. This notice is informational and does
not replace the full license texts included with release archives.
