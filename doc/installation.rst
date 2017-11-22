Installation
============

.. toctree::
   :maxdepth: 1
   :glob:

Anaconda
--------

lsmgridtrack is designed for use with `Anaconda <https://www.continuum.io/downloads>`_ Python from Continuum Analytics. You can install either the full Anaconda package or Miniconda.

.. note::
   lsmgridtrack is only released for Python 3.5 and 3.6. If you install an Anaconda with a Python version that is different, you must follow the installation `In a conda environment`_ instructions. 

To install the current release
------------------------------

lsmgridtrack depends on packages outside of the default conda channels, so we need to add these. The order also matters, because this determines channel priority.

.. code-block:: guess

   conda config --add channels SimpleITK
   conda config --add channels siboles
   conda config --add channels conda-forge

In a conda environment
~~~~~~~~~~~~~~~~~~~~~~

By installing lsmgridtrack into its own conda environment it is guaranteed to not interfere with other anaconda packages and vice versa. Therefore, this approach is the most recommended. This can be accomplished in one command line entry.

.. code-block:: guess

   conda create -n lsm python=3.6 lsmgridtrack

This creates an environment named *lsm* (it can be whatever name you want except existing environment names). The environment will use Python version 3.6. Finally, lsmgridtrack and all of its dependencies will be installed.

To use *lsmgridtrack* this environment will need to be active. In a command terminal, type:

.. code-block:: guess

   activate lsm

on Windows. Or on Linux type:

.. code-block:: guess

   source activate lsm

You should see the command prompt change to include *(lsm)* in parentheses. Again, replace *lsm* with whatever name you chose for your environment.

To change to a different environment type the above command with the new environment name. To deactivate the environment type:

.. code-block:: guess

   deactivate

on Windows, or:

.. code-block:: guess

   source deactivate

on Linux.

In the root environment
~~~~~~~~~~~~~~~~~~~~~~~

Installation into the root environment of your Anaconda installation is discouraged but can be accomplished by opening a command terminal and typing:

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
