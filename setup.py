from __future__ import division, print_function, absolute_import


from numpy.distutils.core import setup, Extension
from numpy.distutils import misc_util as np_misc_util
import os, sys, copy, glob

import versioneer


BASE_PATH = 'gufunctools'
VERSION_FILE_PATH = os.path.join(BASE_PATH, '_version.py')

versioneer.versionfile_source = VERSION_FILE_PATH
versioneer.versionfile_build = VERSION_FILE_PATH
versioneer.tag_prefix = ''
versioneer.parentdir_prefix = 'gufunctools-'


gufunctools_module = Extension(
    'gufunctools._npy_tools',
    sources = []
)

# include all c files in its source directory
NONUMPY_MODULE_SRC = glob.glob(
    os.path.join('modules', 'nonpy_tools', 'src', '*.c')
)

gufunctools_nonumpy_module = Extension(
    'gufunctools._nonpy_tools',
    sources = NONUMPY_MODULE_SRC,
)

EXAMPLES_MODULE_SRC = glob.glob(
    os.path.join('modules', 'examples', 'src', '*.c')
)

gufunctools_examples_module = Extension(
    'gufunctools.examples',
    sources = EXAMPLES_MODULE_SRC,
    **np_misc_util.get_info('npymath')
)

packages = [
    'gufunctools',
#   'gufunctools.npy_dependent',
#   'gufunctools.no_npy_dependent',
#   'gufunctools.tests',
]

ext_modules = [
#    gufunctools_module,
    gufunctools_nonumpy_module,
    gufunctools_examples_module,
]

setup(name='gufunctools',
      version=versioneer.get_version(),
      description='gufunc inspection tools',
      author='Continuum Analytics, Inc.',
      ext_modules=ext_modules,
      packages=packages,
      license='BSD',
      long_description=open('README.rst').read(),
      cmdclass=versioneer.get_cmdclass()
)
