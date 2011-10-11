=================================
 Various ideas and code snippets
=================================

Environment
===========


* Set your python path in ``.bashrc.`` E.g.::

    PYTHONPATH=$HOME/projects/stochastics/spuq

* Check all your files with ``pep8``::

    pep8 -r spuq/linalg/operator.py

  Option ``-r`` should be default, so that messages are repeated (can set
  ``alias pep8="pep8 -r"`` in ``.bashrc``

* Include coverage report for packages::

    nosetests -vv --with-coverage --cover-package spuq spuq/

  or for single packages or modules::

    nosetests -vv -s --with-coverage --cover-package spuq.linalg spuq/linalg/

  (``-vv`` for verbose output, ``-s`` no not capture stdout)

* To not include functions or single lines add comment ``#pragma: no cover``

Code
====

* Don't use private variables (``__``), only if you have good reason
  to, and then it should be documented why.

