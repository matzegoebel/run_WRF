"""A setuptools based setup module adapted from PyPa's sample project.

See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject

We reproduce their license terms here:

Copyright (c) 2016 The Python Packaging Authority (PyPA)

Permission is hereby granted, free of charge, to any person obtaining a copy of
this file (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from setuptools import setup, find_packages
# faster entry points
import fastentrypoints
# To use a consistent encoding
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='run_wrf',
    version='1.0',
    description='This package helps setting up and running idealized WRF experiments',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/matzegoebel/run_wrf',
    author='Matthias Göbel',
    author_email='matthias-goebel@freenet.de',
    keywords='WRF',
    packages=find_packages(),
    install_requires=['pandas', 'xarray', 'scipy', 'matplotlib'],
    extras_require={
        'test': ['pytest'],
    },
    entry_points={'console_scripts':
                  ['run_wrf=run_wrf.launch_jobs:parse_args',
                   'concat_output=run_wrf.tools:concat_output',
                   'get_runtime=run_wrf.tools:get_runtime',
                   'rankfile=run_wrf.sge_funcs:rankfile',
                   'get_hosts=run_wrf.sge_funcs:get_hosts',
                   'get_hosts_set=run_wrf.sge_funcs:get_hosts_set',
                   'check_namelist_value=run_wrf.get_namelist:check_namelist_value']}
)
