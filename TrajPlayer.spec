# -*- mode: python ; coding: utf-8 -*-
import os
import sys
from pathlib import Path

project_root = Path(SPECPATH)
extra_binaries = []
if os.name == 'nt':
    conda_prefixes = [Path(sys.prefix)]
    if os.environ.get('CONDA_PREFIX'):
        conda_prefixes.insert(0, Path(os.environ['CONDA_PREFIX']))
    for conda_prefix in dict.fromkeys(conda_prefixes):
        conda_bin = conda_prefix / 'Library' / 'bin'
        for dll_name in (
            'libcrypto-3-x64.dll',
            'libssl-3-x64.dll',
            'libexpat.dll',
            'ffi.dll',
            'libbz2.dll',
            'liblzma.dll',
            'sqlite3.dll',
        ):
            dll_path = conda_bin / dll_name
            if dll_path.exists():
                extra_binaries.append((str(dll_path), '.'))


a = Analysis(
    [str(project_root / 'app.py')],
    pathex=[str(project_root)],
    binaries=extra_binaries,
    datas=[
        ('DISTRIBUTION_README.txt', '.'),
        ('LICENSE', '.'),
        ('THIRD_PARTY_NOTICES.md', '.'),
    ],
    hiddenimports=[
        'ase.io.trajectory',
        'ase.io.xyz',
        'ase.io.extxyz',
        'ase.io.proteindatabank',
        'ase.io.cif',
        'ase.io.gromacs',
        'MDAnalysis.coordinates.XTC',
        'MDAnalysis.coordinates.TRR',
        'MDAnalysis.topology.GROParser',
        'MDAnalysis.lib.formats.libmdaxdr',
        'MDAnalysis.lib.formats.cython_util',
        'MDAnalysis.lib._transformations',
        'scipy.spatial._ckdtree',
        'scipy.sparse.csgraph._traversal',
        'PySide6.QtOpenGL',
        'PySide6.QtOpenGLWidgets',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'IPython',
        'PIL',
        '_tkinter',
        'contourpy',
        'ipykernel',
        'jedi',
        'jupyter_client',
        'jupyter_core',
        'kiwisolver',
        'matplotlib',
        'nbformat',
        'notebook',
        'parso',
        'tkinter',
        'traitlets',
        'zmq',
    ],
    noarchive=False,
    optimize=1,
)

if os.name == 'nt':
    # Conda's ICU58 DLLs conflict with the ICU API expected by Qt 6.10.
    a.binaries = [
        entry for entry in a.binaries
        if not (
            str(entry[0]).replace('\\', '/').lower().rsplit('/', 1)[-1].startswith('icu')
            and str(entry[0]).lower().endswith('.dll')
        )
    ]

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='TrajPlayer',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='TrajPlayer',
)
