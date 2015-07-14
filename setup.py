 #!/usr/bin/env python
 # -*- coding: utf-8 -*-

import os
from distutils.core import setup
import glob

setup(name='allograph',
      version='0.1',
      license='ISC',
      description='A library for learning handwritten letters from user demonstration',
      author='Alexis Jacq',
      author_email='alexis.jacq@epfl.ch',
      package_dir = {'': 'src'},
      packages=['allograph'],
      data_files=[('share/allograph/letter_model_datasets/uji_pen_chars2', glob.glob("share/letter_model_datasets/uji_pen_chars2/*")),
                  ('share/allograph/letter_model_datasets/alexis_set_for_children', glob.glob("share/letter_model_datasets/alexis_set_for_children/*")),
		  ('share/allograph/letter_model_datasets/uji_pen_subset', glob.glob("share/letter_model_datasets/uji_pen_subset/*")),
                  ('share/doc/allograph', ['AUTHORS', 'LICENSE', 'README.md'])]
      )
