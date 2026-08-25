from setuptools import setup, find_packages
import codecs
from os import path
import io
import re

with io.open("xdispersion/__init__.py", "rt", encoding="utf8") as f:
    version = re.search(r'__version__ = "(.*?)"', f.read()).group(1)

here = path.abspath(path.dirname(__file__))

with codecs.open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='xdispersion',

    version=version,

    description='Relative dispersion of Lagrangian particle pairs.',
    long_description=long_description,
    long_description_content_type='text/markdown',

    url='https://github.com/miniufo/xdispersion',

    author='miniufo',
    author_email='miniufo@163.com',

    license='MIT',

    classifiers=[
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11'
    ],

    keywords='dispersion Lagrangian particle drifter float',

    packages=find_packages(exclude=['docs', 'tests', "data", "notebooks", "pics", "private"]),

    install_requires=[
        "numpy",
        "xarray",
        "dask",
        "tqdm",
        "scipy",
        "xhistogram",
        "mpmath",
    ],
)
