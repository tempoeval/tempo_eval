"""
This file contains the setup for setuptools to distribute
everything as a (PyPI) package.
"""

import glob
import os
from importlib.machinery import SourceFileLoader
from os import path, listdir
from os.path import join, isfile

from setuptools import setup, find_packages

# define version
version = SourceFileLoader('tempo_eval.version',
                           'tempo_eval/version.py').load_module()

# include references bibtext files
package_data = {'tempo_eval': ['*.bib']}

# define the data to be included in the PyPI package
data_files = []
annotation_dirs = glob.glob(join('annotations', '**') + os.sep,
                            recursive=True)
for directory in annotation_dirs:
    files = [join(directory, f) for f in listdir(directory)
             if isfile(join(directory, f)) and not f.startswith('.')]
    if files:
        data_files.append((directory, files))

# requirements
with open('requirements.txt', 'r') as fh:
    requirements = fh.read().splitlines()

# long description
with open(path.join(path.abspath(path.dirname(__file__)), 'README.md'),
          encoding='utf-8') as f:
    long_description = f.read()


# the actual setup routine
setup(name='tempo_eval',
      version=version.version,
      description='Python tempo estimation algorithm evaluation',
      long_description=long_description,
      long_description_content_type='text/markdown',
      author='Tempo Eval',
      author_email='tempo.eval@gmail.com',
      url='https://github.com/tempoeval/tempo_eval',
      download_url='http://github.com/tempoeval/tempo_eval/releases',
      keywords='audio music sound tempo evaluation',
      license='ISC',
      packages=find_packages(exclude=['tests', 'docs']),
      package_data=package_data,
      data_files=data_files,
      exclude_package_data={
          '': ['tests', 'docs']
      },
      entry_points={
          'console_scripts': [
              'tempo_eval = tempo_eval.commands:tempo_eval_command',
              'convert2jams = tempo_eval.commands:convert2jams_command',
          ]
      },
      install_requires=requirements,
      extras_require={
          'docs': ['sphinx >= 2.0.0',
                   'sphinx_rtd_theme',
                   'sphinxcontrib-versioning >= 2.2.1',
                   'sphinx-autodoc-typehints >= 1.6.0'],      
          'tests': ['pytest',
                    'coverage'
                    'pytest-console-scripts']
      },
      classifiers=['Development Status :: 3 - Alpha',
                   'Environment :: Console',
                   'License :: OSI Approved :: ISC License (ISCL)',
                   'Topic :: Multimedia :: Sound/Audio :: Analysis',
                   'Programming Language :: Python :: 3',
                   'Programming Language :: Python :: 3.6',
                   'Programming Language :: Python :: 3.7'],
      zip_safe=False)
