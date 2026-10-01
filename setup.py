from pathlib import Path
import re
from setuptools import find_packages, setup


ROOT = Path(__file__).parent
version = re.search(
    r'__version__ = "(.*?)"',
    (ROOT / 'xdispersion/__init__.py').read_text(encoding='utf-8'),
).group(1)

setup(
    name='xdispersion',
    version=version,
    description='Relative dispersion of Lagrangian particle pairs.',
    long_description=(ROOT / 'README.md').read_text(encoding='utf-8'),
    long_description_content_type='text/markdown',
    url='https://github.com/miniufo/xdispersion',
    author='miniufo',
    author_email='miniufo@163.com',
    license='MIT',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3.13',
    ],
    keywords='dispersion Lagrangian particle drifter float',
    packages=find_packages(exclude=['docs', 'tests', 'data', 'notebooks', 'pics', 'private']),
    install_requires=[
        'numpy', 'xarray', 'dask', 'tqdm', 'scipy', 'xhistogram', 'mpmath', 'numba',
        # `plot` is re-exported from `__init__`, so importing the package at all
        # needs matplotlib (`mpl_toolkits.axes_grid1`).
        'matplotlib',
        # Hard requirement, not just an accelerator: xarray implements
        # `DataArray.ffill` for dask-backed arrays in
        # `duck_array_ops._push`, which does a bare `import bottleneck as bn`.
        # `measures.relative_diffusivity` calls `.ffill('rtime')` on the
        # separation array, so whenever that array is lazy -- i.e. the
        # chunked path (`chunk=<int>`) -- dask's dtype inference surfaces the
        # absent import as "ValueError: `dtype` inference failed in
        # `map_blocks`".  Eager (unchunked) runs never reach that branch.
        'bottleneck',
    ],
    extras_require={
        'extra': ['xrft'],
        # `h5py` is listed explicitly: h5netcdf >= 1.8 declares it only as an
        # optional extra, so `pip install h5netcdf` no longer provides the
        # engine xarray needs to read the test data.
        'test': ['pytest', 'pytest-cov', 'ruff', 'h5netcdf', 'h5py'],
    },
)
