 #!/usr/bin/env python
 # -*- coding: utf-8 -*-

import os
from distutils.core import setup
import glob
import subprocess

setup(name='allograph',
      version='0.1',
      license='ISC',
      description='A library for learning handwritten letters from user demonstration',
      author='Alexis Jacq',
      author_email='alexis.jacq@epfl.ch',
      package_dir = {'': 'src'},
      packages=['allograph'],
      data_files=[('share/allograph/dataset', glob.glob("share/dataset/*")),
                  ('share/doc/allograph', ['AUTHORS', 'LICENSE', 'README.md'])]
      )

# chmod directory
subprocess.call(['chmod', '-R', '777', '/usr/local/share/allograph/dataset'])
