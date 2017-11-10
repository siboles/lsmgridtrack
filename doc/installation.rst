Installation
============

.. toctree::
   :maxdepth: 1
   :glob:

Anaconda
--------

lsmgridtrack is designed for use with `Anaconda <https://www.continuum.io/downloads>`_ Python from Continuum Analytics. You can install either the full Anaconda package or Miniconda.

To install the current release
------------------------------

lsmgridtrack depends on packages outside of the default conda channels, so we need to add these. The order also matters, because this determines channel priority.

.. code-block:: guess

   conda config --add channels SimpleITK
   conda config --add channels siboles
   conda config --add channels conda-forge

.. code-block:: guess

   conda install lsmgridtrack

To install the latest version
-----------------------------

The source code can be downloaded from `GitHub <https://github.com/siboles/lsmgridtrack/archive/master.zip>`_ or if git is installed cloned with:

.. code-block:: guess

   git clone https://github.com/siboles/lsmgridtrack.git

The module can then be installed following the Standard Python instructions below.

Standard Python
---------------

lsmgridtrack can be installed for a standard Python environment from source.

First the following dependencies must be installed:

 - SimpleITK>=1.0
 - vtk>=7.0
 - numpy
 - pyyaml
 - future
 - openpyxl

 .. note::
    These dependencies do not all exist in PyPi, so it is left to the user to find the appropriate versions for their system. It is thus strongly recommended to go with the Anaconda instructions.

Navigate to src/ in the source code directory tree and type:

.. code-block:: guess

   python setup.py install

This may require sudo priviliges in Linux environments depending on where Python is installed. Alternatively, this can be done in a virtual environment.
