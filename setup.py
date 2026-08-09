from __future__ import annotations

from setuptools import Extension, setup

import numpy


setup(
    ext_modules=[
        Extension(
            "trajplayer._trajcore",
            sources=["trajplayer/_trajcore.c"],
            include_dirs=[numpy.get_include()],
            optional=True,
        )
    ]
)
